#!/usr/bin/env python3
"""ADR-063 Q4-TRACE (#663) attribution of one traced Qwen run (T1) by phase and by native pipeline.

  q4_trace_attribution.py TRACE [--windows W]

Phases come from the owning worker's `transaction` records: each transaction's invocation epoch is the epoch
of the first `scope_definition` after it (the same rule as q4_trace_overhead.py). Joint ordinary decode epochs
are stratified by logical width (lane count) and, when the trace carries the joint witness hook, split at the
`forward_input_end` marker into the forward's own commands and the per-lane settle work (sampling, MTP
catch-up), by the reducer's integer join on first-dispatch scope ids.

Per epoch: commands, dispatches, host encode span (first `Created` begin -> last `Submitted` end), completion
tail (last `Submitted` end -> last `CheckedDrain` end) and epoch span. Per native pipeline (the observer's
`pipeline_identity`, a process-local pipeline handle): dispatch counts per phase and joint width, the grid
shapes it ran, and its first site in a joint forward (command index, dispatch ordinal), which maps it to the
encode order in `encode_joint_decode` (without the hook, the whole joint epoch's order).

Without GPU timing records, GPU time per command is not observable: joint layer chains are committed back
to back and checked together at the end of the forward (the completion tail is the GPU work left after
encoding). With the Q4-TRACE-GPU-TIMING opt-in (`native_gpu_timing` records), each epoch also gets its GPU
busy time (union of command GPU intervals), GPU bubbles (gaps inside its GPU envelope) and per-command GPU
time, and joint epochs that match the trace's `joint_chain_plan` (command count and per-index dispatch
signature) get per-chain GPU time with each chain's layer range; any other epoch is counted as
`chain_map_refused`, never mapped. A commit-to-GPU-start launch time is reported only with both paired host
clock samples, with its measured bound and its stated assumption. Per-kernel time and pipeline names are
never in the trace. Read-only, CPU-only.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from f2_gpu_timing import union_ns
from q1_decision_policy import nearest_rank
from q4_trace_verdict import Jsonl, require

JOINT = "joint_ordinary_decode"


def phase_name(operation, width):
    return f"{operation}/logical{width}" if operation == JOINT else operation


def summary(values):
    return {"n": len(values), "sum": sum(values), "median": nearest_rank(values, 0.5),
            "p95": nearest_rank(values, 0.95), "max": max(values, default=None)}


class Attribution:
    def __init__(self):
        self.pending = None              # (operation, width) of the transaction awaiting its first scope
        self.epochs = {}                 # epoch -> {"operation", "width"}
        self.inputs = {}                 # epoch -> joint forward_id (hook trace only)
        self.ends = {}                   # epoch -> (watermark, end_ns)
        self.commands = {}               # live command id -> state
        self.done = defaultdict(list)    # epoch -> retired command states
        self.kernels = defaultdict(lambda: {"dispatches": Counter(), "grids": Counter(), "first_joint_site": None})
        self.kinds = Counter()
        self.first_at = self.last_at = None
        self.plan, self.pairs, self.gpu_seen = None, [], False

    def consume(self, record):
        kind, p = record["kind"], record["payload"]
        self.kinds[kind] += 1
        if kind == "transaction":
            width = len(p["logical_lanes"])
            self.pending = (p["operation"], width)
            self.first_at = record["at_ns"] if self.first_at is None else self.first_at
            self.last_at = record["at_ns"]
        elif kind == "scope_definition" and self.pending is not None:
            self.epochs.setdefault(p["scope"]["epoch"], {"operation": self.pending[0], "width": self.pending[1]})
            self.pending = None
        elif kind == "forward_input" and p.get("source") == "qwen35.joint_decode.actual_slot_rows":
            self.inputs[p["epoch"]] = p["forward_id"]
        elif kind == "forward_input_end":
            self.ends[p["epoch"]] = (p["scope_watermark"], p["end_monotonic_ns"])
        elif kind == "native_event":
            self.event(p)
        elif kind == "joint_chain_plan":
            self.plan = p
        elif kind == "native_gpu_timing":
            c = self.commands.get(p["command_id"])
            available = p["timestamps"].get("Available")
            if c is not None and available is not None:
                c["gpu"] = (available["start_seconds"], available["end_seconds"])
                self.gpu_seen = True
        elif kind in ("native_arm_ack", "native_final") and "clock_pair" in p:
            self.pairs.append(p["clock_pair"])

    def event(self, p):
        identity, kind = p["command_id"], p["kind"]
        if kind == "Created":
            self.commands[identity] = {"epoch": p["invocation_epoch"], "created": p["begin_monotonic_ns"],
                                       "dispatches": [], "first_scope": None, "submit_end": None, "drain_end": None,
                                       "gpu": None}
            return
        c = self.commands.get(identity)
        if c is None:
            return
        if kind == "DispatchEncoded":
            if c["first_scope"] is None and p["scope"] is not None:
                c["first_scope"] = p["scope"]["id"]
            geometry = p["geometry"]
            c["dispatches"].append((geometry["pipeline_identity"], tuple(geometry["grid"]),
                                    tuple(geometry["threads_per_threadgroup"])))
        elif kind == "Submitted":
            c["submit_end"] = p["end_monotonic_ns"]
        elif kind == "CheckedDrain":
            c["drain_end"] = p["end_monotonic_ns"]
        elif kind == "Dropped":
            self.done[c["epoch"]].append(self.commands.pop(identity))

    def part(self, epoch, c):
        """forward | settle | pre_forward | boundary_ambiguous for a hooked joint epoch, else None."""
        if epoch not in self.inputs or epoch not in self.ends:
            return None
        watermark, end_ns = self.ends[epoch]
        if c["first_scope"] is not None:
            if c["first_scope"] < self.inputs[epoch]:
                return "pre_forward"
            return "forward" if c["first_scope"] <= watermark else "settle"
        if c["created"] == end_ns:
            return "boundary_ambiguous"
        return "forward" if c["created"] < end_ns else "settle"

    def finish(self):
        phases = defaultdict(lambda: defaultdict(list))
        parts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        unattributed = Counter()
        for epoch, commands in self.done.items():
            info = self.epochs.get(epoch)
            if info is None:
                unattributed["commands"] += len(commands)
                unattributed["dispatches"] += sum(len(c["dispatches"]) for c in commands)
                continue
            name = phase_name(info["operation"], info["width"])
            self.span(phases[name], commands)
            ordered = sorted(commands, key=lambda c: c["created"])
            labels = [self.part(epoch, c) for c in ordered]
            hooked = epoch in self.inputs and epoch in self.ends
            forward = [c for c, part in zip(ordered, labels) if part == "forward"] if hooked else ordered
            for c, part in zip(ordered, labels):
                label = name if part is None else f"{name}:{part}"
                for ordinal, (pipeline, grid, threads) in enumerate(c["dispatches"], 1):
                    kernel = self.kernels[pipeline]
                    kernel["dispatches"][label] += 1
                    kernel["grids"][(grid, threads)] += 1
                    if (info["operation"] == JOINT and info["width"] == 4 and c in forward
                            and kernel["first_joint_site"] is None):
                        # Site in the forward's encode order: which command of the forward, which dispatch.
                        kernel["first_joint_site"] = {"command_index": forward.index(c), "dispatch_ordinal": ordinal}
            if epoch in self.inputs and epoch in self.ends:
                by_part = defaultdict(list)
                for c in commands:
                    by_part[self.part(epoch, c)].append(c)
                for part, members in by_part.items():
                    self.span(parts[name][part], members)
        chains, refused = self.chains()
        total = sum(sum(c["dispatches"].values()) for c in self.kernels.values())
        kernels = []
        for pipeline, k in sorted(self.kernels.items(), key=lambda item: -sum(item[1]["dispatches"].values())):
            count = sum(k["dispatches"].values())
            kernels.append({"pipeline_identity": pipeline, "dispatches": count,
                            "share": count / total if total else None,
                            "by_phase": dict(sorted(k["dispatches"].items())),
                            "grids": [{"grid": list(g), "threads_per_threadgroup": list(t), "dispatches": n}
                                      for (g, t), n in k["grids"].most_common(3)],
                            "distinct_grids": len(k["grids"]), "first_joint_site": k["first_joint_site"]})
        result = {"schema": "hf2q.q4-trace.attribution.v1",
                "phases": {name: self.render(metrics) for name, metrics in sorted(phases.items())},
                "joint_forward_settle": {name: {part: self.render(m) for part, m in sorted(by.items())}
                                         for name, by in sorted(parts.items())},
                "kernels": kernels, "dispatches_total": total, "unattributed": dict(unattributed),
                "transactions_span_ns": None if self.first_at is None else self.last_at - self.first_at,
                "records_by_kind": dict(sorted(self.kinds.items())),
                "not_observed": ["GPU time per kernel or per command (joint layer chains are committed back to "
                                 "back and checked together at the forward's end)", "pipeline names"]}
        if self.gpu_seen:
            result["not_observed"] = ["GPU time per kernel (a command buffer is the smallest timed unit)",
                                      "pipeline names"]
            result["joint_chains"] = chains
            result["chain_map_refused"] = refused
            result["chain_plan"] = self.plan
            result["launch"] = self.launch()
        return result

    def chains(self):
        """Per-chain GPU time of joint epochs that match the published plan; others are refused."""
        if self.plan is None or not self.gpu_seen:
            return {}, 0
        commands, kinds = self.plan["commands"], self.plan["layer_kinds"]
        signature, refused = {}, 0
        chains = defaultdict(lambda: defaultdict(list))
        for epoch, members in self.done.items():
            info = self.epochs.get(epoch)
            if info is None or info["operation"] != JOINT:
                continue
            ordered = sorted(members, key=lambda c: c["created"])
            counts = [len(c["dispatches"]) for c in ordered]
            expected = signature.setdefault(info["width"], counts) if len(ordered) == len(commands) else None
            if expected != counts or any(c["gpu"] is None for c in ordered):
                refused += 1
                continue
            for label, c in zip(commands, ordered):
                chains[phase_name(JOINT, info["width"])][label].append(round((c["gpu"][1] - c["gpu"][0]) * 1e9))
        def layers(label):
            if not label.startswith("layers"):
                return None
            a, b = (int(x) for x in label[label.index("[") + 1:-1].split(","))
            return {"range": [a, b], "kinds": kinds[a:b]}
        return ({name: [{"command": label, "layers": layers(label), "gpu_ns": summary(by[label])}
                        for label in commands if label in by] for name, by in sorted(chains.items())}, refused)

    def launch(self):
        """Commit end -> GPU start per command, only with both paired host clock samples."""
        if len(self.pairs) < 2:
            return {"reported": False, "reason": "needs the arm and final paired host clock samples"}
        def offset(pair):
            numer, denom = pair["mach_timebase"]
            return pair["monotonic_ns"] - pair["mach_absolute_ticks"] * numer // denom
        first, last = offset(self.pairs[0]), offset(self.pairs[-1])
        bound = max(pair["read_span_ns"] for pair in self.pairs) + abs(last - first)
        phases = defaultdict(list)
        for epoch, members in self.done.items():
            info = self.epochs.get(epoch)
            for c in members:
                if info is not None and c["gpu"] is not None and c["submit_end"] is not None:
                    gpu_start = round(c["gpu"][0] * 1e9) + first
                    phases[phase_name(info["operation"], info["width"])].append(gpu_start - c["submit_end"])
        return {"reported": True, "bound_ns": bound,
                "assumption": "Metal host seconds are mach_absolute_time x timebase; the bound covers the "
                              "sample read spans and the monotonic-mach offset drift only",
                "by_phase": {name: summary(values) for name, values in sorted(phases.items())}}

    @staticmethod
    def span(metrics, commands):
        created = [c["created"] for c in commands]
        submits = [c["submit_end"] for c in commands if c["submit_end"] is not None]
        drains = [c["drain_end"] for c in commands if c["drain_end"] is not None]
        metrics["commands"].append(len(commands))
        gpu = [(round(c["gpu"][0] * 1e9), round(c["gpu"][1] * 1e9)) for c in commands if c["gpu"] is not None]
        if gpu:
            busy = union_ns(gpu)
            metrics["gpu_busy_ns"].append(busy)
            metrics["gpu_bubble_ns"].append(max(b for _, b in gpu) - min(a for a, _ in gpu) - busy)
            metrics["command_gpu_ns"].extend(b - a for a, b in gpu)
        metrics["dispatches"].append(sum(len(c["dispatches"]) for c in commands))
        if submits:
            metrics["encode_span_ns"].append(max(submits) - min(created))
        if submits and drains:
            metrics["completion_tail_ns"].append(max(drains) - max(submits))
            metrics["span_ns"].append(max(drains) - min(created))

    @staticmethod
    def render(metrics):
        return {name: summary(values) for name, values in sorted(metrics.items())}


def attribute(records):
    state = Attribution()
    for record in records:
        state.consume(record)
    return state.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--windows", help="q2_traced_freeze windows of the same run: cross-checks the joint split")
    args = parser.parse_args(argv)
    try:
        result = attribute(Jsonl(args.trace))
        if args.windows:
            joint = json.loads(Path(args.windows).read_text())["validate_trace"].get(JOINT)
            require(type(joint) is dict, "windows report has no joint_ordinary_decode section")
            reduced = Counter(f"{JOINT}/logical{f['logical_rows']}" for f in joint["forwards"])
            mine = {name: m["span_ns"]["n"] for name, by in result["joint_forward_settle"].items()
                    for part, m in by.items() if part == "forward"}
            result["reducer_cross_check"] = {"reducer_forwards": dict(reduced), "attributed_forwards": mine,
                                             "agree": dict(reduced) == mine}
    except (KeyError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
