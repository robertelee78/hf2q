"""ADR-063 Q4-TRACE-GPU-TIMING: per-command GPU timing records of a native diagnostic trace.

Used by f2_native_trace.NativeTrace only when the header's `native_observer.gpu_timestamps` is true.
mlx-native emits one terminal timing event per observed command in the same call as its checked drain,
so in the trace a command's timing follows its `CheckedDrain` and precedes its `Dropped`. Every check here
runs as its record arrives. GPU times are Metal host seconds (mach time); they are compared with the
trace's CLOCK_MONOTONIC instants only as durations, with a tolerance, never as instants.
"""
import math

CLOCK = "MetalCommandBufferHostSecondsSystemMachTime"
TIMING = {"schema_version", "command_id", "invocation_epoch", "queue_identity", "device_registry_id", "timestamps"}
AVAILABLE = {"clock", "start_seconds", "end_seconds"}
UNKNOWN = {"reason", "checked_status", "raw_bits"}
REASONS = {"NotHealthyCompleted", "EmptyCommand", "NonFinite", "NonPositive", "NotStrictlyOrdered",
           "DroppedWithoutCheckedTiming", "TaintedObservation"}
PAIR = {"monotonic_ns", "uptime_raw_ns", "mach_absolute_ticks", "mach_timebase", "read_span_ns"}
PLAN = {"model", "n_layers", "chain_n", "layer_kinds", "commands"}
MIN_TOLERANCE_NS = 1_000


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    return type(value) is int and 0 <= value <= 2**64 - 1


def clock_pair(value):
    need(type(value) is dict and set(value) == PAIR and
         all(number(value[k]) for k in PAIR - {"mach_timebase"}) and
         type(value["mach_timebase"]) is list and len(value["mach_timebase"]) == 2 and
         all(number(x) and 0 < x <= 2**32 - 1 for x in value["mach_timebase"]),
         "closed paired host clock sample")
    return value


def plan_commands(n_layers, chain_n):
    """The joint forward's commands in commit order, by the server's closes_chain rule."""
    commands, start = ["embed"], 0
    for layer in range(n_layers):
        if (layer + 1) % chain_n == 0 or layer + 1 == n_layers:
            commands.append(f"layers[{start},{layer + 1})")
            start = layer + 1
    if start < n_layers:
        commands.append(f"layers.flush[{start},{n_layers})")
    return commands + ["output_head"]


def union_ns(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


class GpuTiming:
    def __init__(self):
        self.pairs, self.plan = [], None
        self.available = self.unknown = 0
        self.reasons = {}

    def tolerance_ns(self):
        return max([MIN_TOLERANCE_NS] + [pair["read_span_ns"] for pair in self.pairs])

    def clock_pair(self, value):
        self.pairs.append(clock_pair(value))

    def joint_chain_plan(self, p):
        need(self.plan is None and type(p) is dict and set(p) == PLAN and p["model"] == "qwen35" and
             number(p["n_layers"]) and 0 < p["n_layers"] <= 1024 and number(p["chain_n"]) and p["chain_n"] > 0 and
             type(p["layer_kinds"]) is list and len(p["layer_kinds"]) == p["n_layers"] and
             all(kind in ("linear", "full") for kind in p["layer_kinds"]) and
             p["commands"] == plan_commands(p["n_layers"], p["chain_n"]),
             "one closed joint chain plan matching the commit rule")
        self.plan = p

    def timing(self, p, c):
        """One timing record for live command state `c` (from its Created event onward)."""
        need(type(p) is dict and set(p) == TIMING and p["schema_version"] == 1 and
             all(p[k] == c["created"][k] for k in ("invocation_epoch", "queue_identity", "device_registry_id")),
             "closed GPU timing record bound to its command")
        need(c["timing"] is None, "duplicate GPU timing for one command")
        stamps = p["timestamps"]
        need(type(stamps) is dict and len(stamps) == 1, "closed GPU timestamps variant")
        (variant, value), = stamps.items()
        if variant == "Available":
            need(set(value) == AVAILABLE and value["clock"] == CLOCK and
                 all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in ("start_seconds", "end_seconds"))
                 and 0 < value["start_seconds"] < value["end_seconds"], "closed Available GPU interval")
            need(c["drain"] is not None and c["drain"]["payload"]["checked_status"] == 4,
                 "Available GPU timing before a healthy checked drain")
            gpu_ns = round((value["end_seconds"] - value["start_seconds"]) * 1e9)
            host_ns = c["drain"]["payload"]["end_monotonic_ns"] - c["submit"]["payload"]["begin_monotonic_ns"]
            need(gpu_ns <= host_ns + self.tolerance_ns(), "GPU duration exceeds its host submit-to-drain bracket")
            c["timing"] = (value["start_seconds"], value["end_seconds"])
            self.available += 1
        else:
            need(variant == "Unknown" and set(value) == UNKNOWN and value["reason"] in REASONS and
                 (value["checked_status"] is None or number(value["checked_status"])) and
                 (value["raw_bits"] is None or (type(value["raw_bits"]) is list and len(value["raw_bits"]) == 2 and
                                                all(number(x) for x in value["raw_bits"]))),
                 "closed Unknown GPU timing")
            reason, drained = value["reason"], c["drain"] is not None
            cause = {"EmptyCommand": not (c["dispatches"] or c["noncompute"]),
                     "DroppedWithoutCheckedTiming": not drained,
                     "NonFinite": drained and value["raw_bits"] is not None,
                     "NonPositive": drained and value["raw_bits"] is not None,
                     "NotStrictlyOrdered": drained and value["raw_bits"] is not None,
                     # The reducer refuses unhealthy drains and tainted events outright.
                     "NotHealthyCompleted": False, "TaintedObservation": False}[reason]
            need(cause, f"GPU timing reason {reason} without its cause")
            c["timing"] = reason
            self.unknown += 1
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @staticmethod
    def dropped(c):
        need(c["timing"] is not None, "command dropped without its GPU timing record")

    def finish(self, commands):
        """Per-epoch clock-free bound and per-queue commit-order inversions over retired commands."""
        epochs, queues = {}, {}
        for c in commands:
            epochs.setdefault(c["created"]["invocation_epoch"], []).append(c)
            if type(c["timing"]) is tuple and c["submit"]:
                queues.setdefault(c["queue"], []).append((c["submit"]["payload"]["end_monotonic_ns"], c["timing"][0]))
        checked = 0
        for members in epochs.values():
            gpu = [(round(c["timing"][0] * 1e9), round(c["timing"][1] * 1e9))
                   for c in members if type(c["timing"]) is tuple]
            drains = [c["drain"]["payload"]["end_monotonic_ns"] for c in members if c["drain"]]
            if not gpu or not drains:
                continue
            host = max(drains) - min(c["created"]["begin_monotonic_ns"] for c in members)
            need(union_ns(gpu) <= host + self.tolerance_ns(), "epoch GPU busy time exceeds its host span")
            checked += 1
        inversions = 0
        for starts in queues.values():
            ordered = [start for _, start in sorted(starts)]
            inversions += sum(1 for a, b in zip(ordered, ordered[1:]) if b < a)
        return {"available": self.available, "unknown": self.unknown, "unknown_by_reason": dict(sorted(self.reasons.items())),
                "epochs_bounded": checked, "commit_order_inversions": inversions,
                "clock_pairs": len(self.pairs), "tolerance_ns": self.tolerance_ns(),
                "joint_chain_plan": self.plan}
