"""Synthetic negative controls of the actual diagnostic validator entrypoints.

These fixtures are not hardware evidence or a production-client capture.
"""
import copy
import unittest

import f2_trace as trace


def fixture():
    identity = {name: "a" * (40 if name == "server_source" else 64) for name in
                ("server_source", "server_binary_sha256", "native_checksum", "model_sha256",
                 "tokenizer_sha256", "template_sha256", "profile_sha256")}
    identity["native_version"] = "0.15.8"
    policy = {"schema": "hf2q.f2.forward-witness-policy.v1", "operator": "target_input_rms_norm",
              "input_sources": ["qwen35.greedy.actual_tokens"],
              "coverage": "each_admitted_lane_in_first_fixed_window",
              "batching_scope": "operator_only_full_body_unknown"}
    native = {"event_limit": 1000, "metadata_limit_bytes": 1024**2, "diagnostic_case": True,
              "control_nonce": "n" * 32, "witness_manifest": policy}
    header = {"identity": identity, "run_id": "synthetic_native_control", "server_budget_bytes": 128 * 1024**2,
              "client_budget_bytes": 128 * 1024**2, "server_jsonl_budget_bytes": 126 * 1024**2,
              "clock": "CLOCK_MONOTONIC nanoseconds since origin", "clock_id": "CLOCK_MONOTONIC",
              "origin_monotonic_ns": 1000, "config_sha256": "c" * 64, "native_observer": native}
    scenario = {"schema": "hf2q.f2.trace-expectation.v1", "run_id": header["run_id"],
                "config_sha256": header["config_sha256"], "forward_witness_manifest": copy.deepcopy(policy)}
    records = []
    def add(kind, payload, at):
        records.append({"schema": trace.SCHEMA, "seq": len(records), "kind": kind, "at_ns": at,
                        "payload": copy.deepcopy(payload)})
    status = {"installed": True, "invalid": False, "events": 0, "unknown_scopes": 0,
              "event_limit": 1000, "metadata_reserved_bytes": 0, "metadata_limit_bytes": 1024**2,
              "dropped_events": 0, "unobserved_commits": 7, "capture_valid": True,
              "comparison_qualified": False, "pending_commands": 0, "pending_command_limit": 1024}
    row = {"request_id": 5, "slot_id": 0, "generation": 7, "token_position": 123,
           "token_count": 1, "phase": "ordinary"}
    owner = {"request_id": 5, "slot_id": 0, "generation": 7, "phase": "decode", "active": True,
             "forward_eligibility": True, "runnable_forward": True, "runnable_emit": False, "inflight": False, "blocked": None}
    scope = {"epoch": 10, "id": 20}
    add("header", header, 0)
    add("native_baseline", status, 1)
    add("native_arm_ack", {"nonce": "n" * 32, "clock_id": "CLOCK_MONOTONIC", "worker_queue_checked": True,
        "quiescent": True, "baseline": status, "scope": "one_fresh_phase_diagnostic"}, 2)
    add("slots", {"cause": "synthetic control only", "slots": [owner]}, 3)
    add("forward_input", {"forward_id": 30, "epoch": 10, "source": "qwen35.greedy.actual_tokens",
        "role": "target", "matrix_rows": 1, "packed_rows": [{"matrix_row_start": 0,
        "matrix_row_end": 1, "logical": row}], "operator_site": "pre_attention_input",
        "execution_proven": False}, 4)
    add("scope_definition", {"scope": scope, "operation": "target.input_rms_norm",
        "membership": "Known", "rows": [row]}, 5)
    add("forward_witness", {"forward_id": 30, "scope": scope, "source": "qwen35.greedy.actual_tokens",
        "operator": "target_input_rms_norm", "layer": None, "role": "target", "rows": 1, "columns": 8,
        "dtype": "F32", "input_parent_identity": 12345, "input_byte_offset": 0, "input_read_bytes": 32,
        "input_data_bytes": 32, "input_parent_bytes": 32, "dispatches_expected": 1,
        "operator_site": "pre_attention_input",
        "execution_proven": False, "full_body_batching_proven": False}, 6)
    event = {"clock_id": "CLOCK_MONOTONIC", "command_id": 1, "invocation_epoch": 10,
        "queue_identity": 50, "device_registry_id": 60, "kind": "Created", "begin_monotonic_ns": 1008,
        "end_monotonic_ns": 1008, "dispatch_ordinal": 0, "noncompute_operations": 0, "scope": None,
        "captured_scope": None, "capture_command_id": None, "capture_generation": None,
        "replay_binding": "Direct", "tainted": False, "geometry": None, "submitted": False,
        "checked_status": None}
    add("native_event", event, 8)
    event.update(kind="DispatchEncoded", begin_monotonic_ns=1009, end_monotonic_ns=1009,
        dispatch_ordinal=1, scope=scope, geometry={"pipeline_identity": 70, "dispatch_kind": "thread_groups",
        "grid": [1, 1, 1], "threads_per_threadgroup": [32, 1, 1]})
    add("native_event", event, 9)
    event.update(kind="Submitted", begin_monotonic_ns=1020, end_monotonic_ns=1022,
                 scope=None, geometry=None, submitted=True)
    add("native_event", event, 22)
    event.update(kind="CheckedDrain", begin_monotonic_ns=2_000_001_004,
                 end_monotonic_ns=2_000_001_005, checked_status=4)
    add("native_event", event, 2_000_000_005)
    event.update(kind="Dropped", begin_monotonic_ns=2_000_001_006, end_monotonic_ns=2_000_001_006)
    add("native_event", event, 2_000_000_006)
    status["events"] = 5
    add("native_final", status, 2_000_000_007)
    add("end", {"error": None, "dropped_events": 0}, 2_000_000_008)
    return records, identity, scenario


def record(records, kind, native_kind=None):
    return next(item for item in records if item["kind"] == kind and
                (native_kind is None or item["payload"]["kind"] == native_kind))


def packed_quad(records, scenario, source):
    for policy in (records[0]["payload"]["native_observer"]["witness_manifest"],
                   scenario["forward_witness_manifest"]):
        policy["input_sources"] = [source]
    owner = record(records, "slots")["payload"]["slots"][0]
    record(records, "slots")["payload"]["slots"] = [dict(owner, request_id=5+i, slot_id=i) for i in range(4)]
    one = record(records, "forward_input")["payload"]["packed_rows"][0]["logical"]
    logical = [dict(one, request_id=5+i, slot_id=i) for i in range(4)]
    declaration = record(records, "forward_input")["payload"]
    declaration.update(source=source, matrix_rows=4, packed_rows=[{
        "matrix_row_start": i, "matrix_row_end": i+1, "logical": row} for i, row in enumerate(logical)])
    record(records, "scope_definition")["payload"]["rows"] = logical
    record(records, "forward_witness")["payload"].update(source=source, rows=4,
        input_read_bytes=128, input_data_bytes=128, input_parent_bytes=128)
    record(records, "native_event", "DispatchEncoded")["payload"]["geometry"]["grid"] = [4, 1, 1]


class NativeControls(unittest.TestCase):
    def test_missing_or_unretired_native_owners_cannot_qualify(self):
        for kind in ("native_baseline", "native_arm_ack", "native_final"):
            for change in ("missing", "pending", "limit", "bool"):
                with self.subTest(kind=kind, change=change):
                    records, identity, scenario = fixture()
                    value = next(r["payload"] for r in records if r["kind"] == kind)
                    if kind == "native_arm_ack":
                        value = value["baseline"]
                    if change == "missing":
                        del value["pending_commands"]
                    elif change == "pending":
                        value["pending_commands"] = 1
                    elif change == "limit":
                        value["pending_command_limit"] = 0
                    else:
                        value["pending_commands"] = False
                    with self.assertRaises(ValueError):
                        trace.validate_trace(records, identity, scenario)

    def test_actual_packed_quad_is_operator_quad_not_full_body_claim(self):
        records, identity, scenario = fixture()
        packed_quad(records, scenario, "gemma4.decode.actual_token_slot_position_arrays")
        result = trace.validate_trace(records, identity, scenario)
        self.assertTrue(result["operator_shared_batch4_observed"])
        self.assertFalse(result["shared_batch4_claim"])
        self.assertEqual(trace.qualify_window(records, (0, 4), 0)["status"], "Qualified")

    def test_scalar_source_cannot_be_relabelled_as_packed_quad(self):
        records, identity, scenario = fixture()
        packed_quad(records, scenario, "qwen35.greedy.actual_tokens")
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)

    def test_actual_entrypoint_accepts_checked_late_drain_and_operator_scope_only(self):
        records, identity, scenario = fixture()
        result = trace.validate_trace(records, identity, scenario)
        self.assertTrue(result["qualified"])
        self.assertFalse(result["shared_batch4_claim"])
        self.assertEqual(result["full_body_batching"], "Unknown")
        self.assertEqual(trace.qualify_window(records, (0, 1), 0)["status"], "Qualified")

    def test_required_witness_cannot_be_replaced_by_well_formed_auxiliary(self):
        records, identity, scenario = fixture()
        scope = record(records, "scope_definition")["payload"]
        scope.update(operation="opaque_auxiliary", membership="OpaqueAuxiliary", rows=[])
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)

    def test_raw_escape_or_unobserved_commit_cannot_be_laundered(self):
        for mutate in (lambda r: record(r, "native_event", "Submitted")["payload"].update(tainted=True),
                       lambda r: record(r, "native_final")["payload"].update(unobserved_commits=8)):
            records, identity, scenario = fixture()
            mutate(records)
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_stale_owner_or_one_row_substitution_is_not_physical_participation(self):
        for mutate in (lambda r: record(r, "forward_input")["payload"]["packed_rows"][0]["logical"].update(generation=8),
                       lambda r: record(r, "forward_witness")["payload"].update(rows=4)):
            records, identity, scenario = fixture()
            mutate(records)
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_missing_drain_or_lying_native_counter_refuses(self):
        for mutate in (lambda r: record(r, "native_event", "CheckedDrain")["payload"].update(checked_status=5),
                       lambda r: record(r, "native_final")["payload"].update(events=4)):
            records, identity, scenario = fixture()
            mutate(records)
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_private_manifest_is_frozen_externally_not_selected_by_trace(self):
        records, identity, scenario = fixture()
        scenario["forward_witness_manifest"]["operator"] = "opaque_auxiliary"
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)

    def test_forward_witness_requires_current_capture_mapping_and_exact_geometry(self):
        for mutate in (lambda p: p.update(invocation_epoch=11),
                       lambda p: p["geometry"].update(grid=[0, 1, 1])):
            records, identity, scenario = fixture()
            mutate(record(records, "native_event", "DispatchEncoded")["payload"])
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_counter_reset_or_missing_arm_refuses(self):
        records, identity, scenario = fixture()
        record(records, "native_arm_ack")["payload"]["nonce"] = "x" * 32
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)

    def test_forward_demand_v2_preserves_ready_metric_and_requires_actual_service(self):
        records, identity, scenario = fixture()
        busy = copy.deepcopy(record(records, "slots"))
        busy["at_ns"] = 10
        busy["payload"]["slots"][0].update(runnable_forward=False,
            runnable_emit=False, inflight=None, blocked="OwningWorkerTransaction")
        records.insert(9, busy)  # after actual encoded witness, before its commit
        for seq,item in enumerate(records):
            item["seq"] = seq
        trace.validate_trace(records, identity, scenario)
        self.assertEqual(trace.qualify_window(records, (0,1), 0)["status"], "Unqualified")
        result = trace.qualify_window(records, (0,1), 0, population_policy="forward_demand_v2")
        self.assertEqual(result["status"], "Qualified")
        self.assertLess(result["ready_now_fraction"], 0.001)
        self.assertGreater(result["target_fraction"], 0.99)
        self.assertIn("never Running", result["service_semantics"])

    def test_cancellation_terminal_error_or_unknown_eligibility_ends_service_credit(self):
        for reason,eligible in (("ClientClosed",False),("RequestTerminal",False),
                                ("Cancelled",False),("WorkerError",False),
                                ("EligibilityUnknown",None)):
            records, identity, scenario = fixture()
            stopped = copy.deepcopy(record(records,"slots"))
            stopped["at_ns"] = 100
            stopped["payload"]["slots"][0].update(runnable_forward=False,
                forward_eligibility=eligible,blocked=reason)
            records.insert(10,stopped)
            for seq,item in enumerate(records): item["seq"] = seq
            trace.validate_trace(records,identity,scenario)
            result = trace.qualify_window(records,(0,1),0,population_policy="forward_demand_v2")
            self.assertEqual(result["status"],"Unqualified",reason)
            self.assertLess(result["target_fraction"],0.001)

    def test_terminal_reason_cannot_hide_behind_true_eligibility_or_resume_same_generation(self):
        records,identity,scenario = fixture()
        owner = record(records,"slots")["payload"]["slots"][0]
        owner["blocked"] = "ClientClosed"
        with self.assertRaises(ValueError): trace.validate_trace(records,identity,scenario)
        records,identity,scenario = fixture()
        first = record(records,"slots")
        first["payload"]["slots"][0].update(forward_eligibility=False,
            runnable_forward=False,blocked="Cancelled")
        revived = copy.deepcopy(first)
        revived["at_ns"] = 4
        revived["payload"]["slots"][0].update(forward_eligibility=True,blocked=None)
        records.insert(4,revived)
        for seq,item in enumerate(records): item["seq"] = seq
        with self.assertRaises(ValueError): trace.validate_trace(records,identity,scenario)

    def test_inflight_boolean_cannot_replace_native_forward_service_or_progress(self):
        records, identity, scenario = fixture()
        owner = record(records, "slots")["payload"]["slots"][0]
        owner.update(runnable_forward=False, inflight=True)
        records[-1]["at_ns"] += 100  # entire first service-selected window retained
        trace.validate_trace(records, identity, scenario)
        result = trace.qualify_window(records, (0,1), 0, population_policy="forward_demand_v2")
        # The first eligible instant is commit-END: its earlier commit bracket
        # is not new work inside that window. An old outstanding heartbeat
        # cannot substitute for the separately required new-forward progress.
        self.assertEqual(result["status"], "Unqualified")
        self.assertTrue(result["slots_without_required_forward_witness"])
        with self.assertRaises(ValueError):
            trace.qualify_window(records, (0,1), 0, population_policy="silent_reinterpretation")

    def test_whole_actual_submit_bracket_must_be_inside_window(self):
        records, identity, scenario = fixture()
        submitted = record(records, "native_event", "Submitted")
        submitted["at_ns"] = 2_000_000_004
        submitted["payload"]["end_monotonic_ns"] = 2_000_001_004
        trace.validate_trace(records, identity, scenario)
        self.assertEqual(trace.qualify_window(records, (0, 1), 0)["status"], "Unqualified")

    def test_temporal_four_rows_of_one_request_are_not_four_agents(self):
        records, identity, scenario = fixture()
        for policy in (records[0]["payload"]["native_observer"]["witness_manifest"],
                       scenario["forward_witness_manifest"]):
            policy["input_sources"] = ["qwen35.forward.actual_tokens"]
        source = record(records, "forward_input")["payload"]
        source["source"] = "qwen35.forward.actual_tokens"
        source["role"] = "target_verify"
        record(records, "forward_witness")["payload"]["source"] = source["source"]
        record(records, "forward_witness")["payload"]["role"] = source["role"]
        source["matrix_rows"] = source["packed_rows"][0]["matrix_row_end"] = 4
        source["packed_rows"][0]["logical"].update(token_count=4, phase="verify")
        record(records, "scope_definition")["payload"]["rows"][0].update(token_count=4, phase="verify")
        record(records, "forward_witness")["payload"].update(rows=4, input_read_bytes=128,
            input_data_bytes=128, input_parent_bytes=128)
        record(records, "native_event", "DispatchEncoded")["payload"]["geometry"]["grid"] = [4, 1, 1]
        result = trace.validate_trace(records, identity, scenario)
        self.assertFalse(result["operator_shared_batch4_observed"])
        self.assertFalse(result["shared_batch4_claim"])

    def test_four_known_singletons_qualify_progress_without_fabricated_physical_quad(self):
        original, identity, scenario = fixture()
        records = copy.deepcopy(original[:4])
        owner = records[3]["payload"]["slots"][0]
        records[3]["payload"]["slots"] = []
        def append(item):
            item["seq"] = len(records)
            records.append(item)
        for i in range(4):
            slot = copy.deepcopy(owner)
            slot.update(request_id=5+i, slot_id=i)
            records[3]["payload"]["slots"].append(slot)
            for old in original[4:10]:  # actual declaration/create/encode/submit prefix
                item = copy.deepcopy(old)
                item["at_ns"] += i * 100
                p = item["payload"]
                if item["kind"] == "forward_input":
                    p.update(forward_id=30+i, epoch=10+i)
                    p["packed_rows"][0]["logical"].update(request_id=5+i, slot_id=i)
                elif item["kind"] == "scope_definition":
                    p["scope"].update(epoch=10+i, id=20+i)
                    p["rows"][0].update(request_id=5+i, slot_id=i)
                elif item["kind"] == "forward_witness":
                    p["forward_id"] = 30+i
                    p["scope"].update(epoch=10+i, id=20+i)
                else:
                    p.update(command_id=1+i, invocation_epoch=10+i)
                    p["begin_monotonic_ns"] += i * 100
                    p["end_monotonic_ns"] += i * 100
                    if p["scope"]:
                        p["scope"].update(epoch=10+i, id=20+i)
                append(item)
        for i in range(4):
            for old in original[10:12]:
                item = copy.deepcopy(old)
                item["at_ns"] += i * 10
                p = item["payload"]
                p.update(command_id=1+i, invocation_epoch=10+i)
                p["begin_monotonic_ns"] += i * 10
                p["end_monotonic_ns"] += i * 10
                append(item)
        final = copy.deepcopy(original[12])
        final["payload"]["events"] = 20
        final["at_ns"] = records[-1]["at_ns"] + 1
        append(final)
        footer = copy.deepcopy(original[13])
        footer["at_ns"] = final["at_ns"] + 1
        append(footer)
        result = trace.validate_trace(records, identity, scenario)
        self.assertFalse(result["operator_shared_batch4_observed"])
        self.assertEqual(trace.qualify_window(records, (0, 4), 0)["status"], "Qualified")


if __name__ == "__main__":
    unittest.main()
