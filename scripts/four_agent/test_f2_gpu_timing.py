"""ADR-063 Q4-TRACE-GPU-TIMING synthetic controls: per-command GPU timing records in the native trace.

Timing records follow mlx-native's per-command causal order (CheckedDrain, timing, Dropped). Not hardware evidence.
"""
import copy
import unittest

import f2_gpu_timing as timing
import f2_trace as trace
from test_f2_native_trace import epoch_commands, four_lanes, record
from test_f2_native_trace_joint import widths

PAIR = {"monotonic_ns": 1_000, "uptime_raw_ns": 900, "mach_absolute_ticks": 21_600, "mach_timebase": [125, 3],
        "read_span_ns": 250}


def gpu_trace(build=four_lanes, available=lambda command, bracket_ns: 0.5, extra=()):
    """`build()` with the opt-in, clock pairs and one timing record after each CheckedDrain.
    `available(command, bracket_ns)` is the GPU duration as a fraction of the host bracket, or an Unknown dict."""
    records, identity, scenario = build()
    if extra:
        epoch_commands(records, extra)
    records[0]["payload"]["native_observer"]["gpu_timestamps"] = True
    record(records, "native_arm_ack")["payload"]["clock_pair"] = dict(PAIR)
    submits, drained, out, start = {}, set(), [], 100.0
    for item in records:
        p = item["payload"]
        if item["kind"] == "native_event" and p["kind"] == "Dropped" and p["command_id"] not in drained:
            # Never checked: mlx-native emits its terminal Unknown timing before the drop.
            out.append({"schema": trace.SCHEMA, "seq": 0, "kind": "native_gpu_timing", "at_ns": item["at_ns"],
                        "payload": {"schema_version": 1, "command_id": p["command_id"],
                                    "invocation_epoch": p["invocation_epoch"], "queue_identity": p["queue_identity"],
                                    "device_registry_id": p["device_registry_id"], "timestamps": {"Unknown": {
                                        "reason": "DroppedWithoutCheckedTiming", "checked_status": None,
                                        "raw_bits": None}}}})
        out.append(item)
        if item["kind"] != "native_event":
            continue
        if p["kind"] == "Submitted":
            submits[p["command_id"]] = p["begin_monotonic_ns"]
        if p["kind"] == "CheckedDrain":
            drained.add(p["command_id"])
            bracket = p["end_monotonic_ns"] - submits[p["command_id"]]
            value = available(p["command_id"], bracket)
            stamps = value if isinstance(value, dict) else {"Available": {
                "clock": timing.CLOCK, "start_seconds": start, "end_seconds": start + bracket * value / 1e9}}
            start += bracket / 1e9 + 1e-3
            out.append({"schema": trace.SCHEMA, "seq": 0, "kind": "native_gpu_timing", "at_ns": item["at_ns"],
                        "payload": {"schema_version": 1, "command_id": p["command_id"],
                                    "invocation_epoch": p["invocation_epoch"], "queue_identity": p["queue_identity"],
                                    "device_registry_id": p["device_registry_id"], "timestamps": stamps}})
    final = record(out, "native_final")
    final["payload"]["events"] += sum(r["kind"] == "native_gpu_timing" for r in out)
    final["payload"]["clock_pair"] = dict(PAIR, monotonic_ns=9_000)
    for seq, item in enumerate(out):
        item["seq"] = seq
    return out, identity, scenario


def renumber(records):
    for seq, item in enumerate(records):
        item["seq"] = seq
    return records


def timings(records):
    return [r for r in records if r["kind"] == "native_gpu_timing"]


class GpuTiming(unittest.TestCase):
    def test_one_available_timing_per_command_reduces_with_bounds(self):
        records, identity, scenario = gpu_trace()
        report = trace.validate_trace(records, identity, scenario)
        self.assertEqual(report["gpu_timing"]["available"], 4)
        self.assertEqual(report["gpu_timing"]["unknown"], 0)
        self.assertEqual(report["gpu_timing"]["epochs_bounded"], 4)
        self.assertEqual(report["gpu_timing"]["commit_order_inversions"], 0)
        self.assertEqual(report["gpu_timing"]["clock_pairs"], 2)
        plain = trace.validate_trace(*four_lanes())
        self.assertEqual(set(report) - set(plain), {"gpu_timing"})
        self.assertEqual(len(report["submissions"]), len(plain["submissions"]))

    def test_joint_trace_with_its_chain_plan(self):
        records, identity, scenario = gpu_trace(widths)
        plan = {"model": "qwen35", "n_layers": 8, "chain_n": 4, "layer_kinds": ["linear"] * 3 + ["full"] +
                ["linear"] * 3 + ["full"], "commands": ["embed", "layers[0,4)", "layers[4,8)", "output_head"]}
        at = record(records, "transaction")
        records.insert(records.index(at) + 1, dict(copy.deepcopy(at), kind="joint_chain_plan", payload=plan))
        report = trace.validate_trace(renumber(records), identity, scenario)
        self.assertEqual(report["gpu_timing"]["joint_chain_plan"], plan)
        self.assertEqual(report["joint_ordinary_decode"]["witnessed"], 4)
        for commands in (["embed", "layers[0,8)", "output_head"], ["layers[0,4)", "layers[4,8)", "output_head"]):
            with self.subTest(commands=commands):
                bad = copy.deepcopy(records)
                record(bad, "joint_chain_plan")["payload"]["commands"] = commands
                with self.assertRaisesRegex(ValueError, "joint chain plan"):
                    trace.validate_trace(bad, identity, scenario)

    def test_plan_rule_matches_the_server(self):
        self.assertEqual(len(timing.plan_commands(64, 4)), 18)
        self.assertEqual(timing.plan_commands(10, 4),
                         ["embed", "layers[0,4)", "layers[4,8)", "layers[8,10)", "output_head"])

    def test_records_need_the_opt_in_and_its_clock_pairs(self):
        records, identity, scenario = gpu_trace()
        del records[0]["payload"]["native_observer"]["gpu_timestamps"]
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)
        for change, reason in ((lambda r: r[0]["payload"]["native_observer"].update(gpu_timestamps=False),
                                "closed separate native diagnostic configuration"),
                               (lambda r: record(r, "native_arm_ack")["payload"].pop("clock_pair"),
                                "arm acknowledgement"),
                               (lambda r: record(r, "native_final")["payload"].pop("clock_pair"),
                                "needs its clock pair"),
                               (lambda r: record(r, "native_arm_ack")["payload"]["clock_pair"].update(extra=1),
                                "closed paired host clock sample")):
            with self.subTest(reason=reason):
                records, identity, scenario = gpu_trace()
                change(records)
                with self.assertRaisesRegex(ValueError, reason):
                    trace.validate_trace(records, identity, scenario)
        records, identity, scenario = four_lanes()
        records.insert(5, {"schema": trace.SCHEMA, "seq": 0, "kind": "native_gpu_timing", "at_ns": 4,
                           "payload": timings(gpu_trace()[0])[0]["payload"]})
        with self.assertRaisesRegex(ValueError, "requires the gpu_timestamps diagnostic"):
            trace.validate_trace(renumber(records), identity, scenario)

    def test_streaming_order_is_drain_then_timing_then_drop_exactly_once(self):
        def before_drain(records):
            item = timings(records)[0]
            records.remove(item)
            drain = next(r for r in records if r["kind"] == "native_event" and r["payload"]["kind"] == "CheckedDrain")
            records.insert(records.index(drain), item)
        def after_drop(records):
            item = copy.deepcopy(timings(records)[0])
            drop = next(r for r in records if r["kind"] == "native_event" and r["payload"]["kind"] == "Dropped")
            item["at_ns"] = drop["at_ns"]
            records.insert(records.index(drop) + 1, item)
        def duplicate(records):
            item = timings(records)[0]
            records.insert(records.index(item) + 1, copy.deepcopy(item))
        def missing(records):
            records.remove(timings(records)[0])
            record(records, "native_final")["payload"]["events"] -= 1
        for mutate, reason in ((before_drain, "before a healthy checked drain"), (after_drop, "unknown or retired"),
                               (duplicate, "duplicate GPU timing"), (missing, "dropped without its GPU timing")):
            with self.subTest(reason=reason):
                records, identity, scenario = gpu_trace()
                mutate(records)
                with self.assertRaisesRegex(ValueError, reason):
                    trace.validate_trace(renumber(records), identity, scenario)

    def test_closed_schema_and_reason_to_cause(self):
        def stamps(value):
            return lambda command, bracket: value if command == 2 else 0.5
        cases = [
            (stamps({"Available": {"clock": "Monotonic", "start_seconds": 1.0, "end_seconds": 1.5}}),
             "closed Available GPU interval"),
            (stamps({"Available": {"clock": timing.CLOCK, "start_seconds": 1.5, "end_seconds": 1.5}}),
             "closed Available GPU interval"),
            (stamps({"Unknown": {"reason": "EmptyCommand", "checked_status": 4, "raw_bits": None, "x": 1}}),
             "closed Unknown GPU timing"),
            (stamps({"Unknown": {"reason": "EmptyCommand", "checked_status": 4, "raw_bits": None}}),
             "EmptyCommand without its cause"),
            (stamps({"Unknown": {"reason": "NotHealthyCompleted", "checked_status": 5, "raw_bits": None}}),
             "NotHealthyCompleted without its cause"),
            (stamps({"Unknown": {"reason": "DroppedWithoutCheckedTiming", "checked_status": None,
                                 "raw_bits": None}}), "DroppedWithoutCheckedTiming without its cause"),
            (stamps({"Other": {}}), "closed Unknown GPU timing"),
        ]
        for available, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    trace.validate_trace(*gpu_trace(available=available))
        ok = stamps({"Unknown": {"reason": "NonFinite", "checked_status": 4, "raw_bits": [1, 2]}})
        report = trace.validate_trace(*gpu_trace(available=ok))
        self.assertEqual(report["gpu_timing"]["unknown_by_reason"], {"NonFinite": 1})

    def test_clock_free_duration_bounds(self):
        with self.assertRaisesRegex(ValueError, "exceeds its host submit-to-drain bracket"):
            trace.validate_trace(*gpu_trace(available=lambda command, bracket: 1.01))
        # Exactly the bracket plus less than the tolerance still passes (float rounding).
        report = trace.validate_trace(*gpu_trace(available=lambda command, bracket: 1 + 100 / bracket))
        self.assertEqual(report["gpu_timing"]["available"], 4)
        # Two commands of one epoch with overlapping host brackets cannot both run their whole bracket
        # back to back on the GPU: the epoch's GPU busy time would exceed its host span.
        with self.assertRaisesRegex(ValueError, "epoch GPU busy time exceeds its host span"):
            trace.validate_trace(*gpu_trace(available=lambda command, bracket: 1.0,
                                            extra=[(9, 900_000_000, 1_900_000_000)]))

    def test_event_count_includes_timing_records(self):
        records, identity, scenario = gpu_trace()
        record(records, "native_final")["payload"]["events"] -= 1
        with self.assertRaisesRegex(ValueError, "native count"):
            trace.validate_trace(records, identity, scenario)

    def test_union_arithmetic(self):
        self.assertEqual(timing.union_ns([(0, 10), (5, 12), (20, 25)]), 17)
        self.assertEqual(timing.union_ns([]), 0)


if __name__ == "__main__":
    unittest.main()
