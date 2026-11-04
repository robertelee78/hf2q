#!/usr/bin/env python3
"""ADR-063 Q4-TRACE (#663) overhead extractor (design §5.3): one rule for both traced arms.

  q4_trace_overhead.py LEFT_TRACE RIGHT_TRACE [--left-windows W] [--left-summary S --right-summary S]

LEFT is the hook arm (T1), RIGHT the parent arm (T0); deltas are LEFT - RIGHT. A joint turn's epoch is
the epoch of the first `scope_definition` after its `joint_ordinary_decode` transaction (T0 has no joint
input, so the input cannot identify it); its logical width is the transaction's lane count. Its span is
the epoch's first native `Created` begin to its last `CheckedDrain` end. Per width: median and p95 per
arm, and their deltas with a seeded bootstrap 95% CI over forwards. Also record counts by kind and
trace bytes per arm. Optional: T1's forward/settle split from its q2_traced_freeze windows report, and
the descriptive makespan delta from the two client summaries (one run per arm). Read-only, CPU-only.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from q1_decision_policy import nearest_rank
from q4_trace_verdict import Jsonl, require

RESAMPLES = 2000
SEED = 663


def scan(records):
    """One streaming pass: joint epoch -> width, per-epoch native span, record counts by kind."""
    kinds, widths, created, drained = Counter(), {}, {}, {}
    pending = None
    for record in records:
        kind, p = record["kind"], record["payload"]
        kinds[kind] += 1
        if kind == "transaction":
            pending = len(p["logical_lanes"]) if p["operation"] == "joint_ordinary_decode" else None
        elif kind == "scope_definition" and pending is not None:
            widths[p["scope"]["epoch"]] = pending
            pending = None
        elif kind == "native_event":
            epoch = p["invocation_epoch"]
            if p["kind"] == "Created":
                created[epoch] = min(created.get(epoch, p["begin_monotonic_ns"]), p["begin_monotonic_ns"])
            elif p["kind"] == "CheckedDrain":
                drained[epoch] = max(drained.get(epoch, p["end_monotonic_ns"]), p["end_monotonic_ns"])
    spans, incomplete = {}, 0
    for epoch, width in sorted(widths.items()):
        if epoch in created and epoch in drained:
            spans.setdefault(width, []).append(drained[epoch] - created[epoch])
        else:
            incomplete += 1
    return {"joint_epochs": len(widths), "incomplete_epochs": incomplete, "spans_ns": spans,
            "records_by_kind": dict(sorted(kinds.items()))}


def stats(values):
    return {"n": len(values), "median_ns": nearest_rank(values, 0.5), "p95_ns": nearest_rank(values, 0.95)}


def bootstrap(left, right, q, rng):
    deltas = sorted(nearest_rank(rng.choices(left, k=len(left)), q) - nearest_rank(rng.choices(right, k=len(right)), q)
                    for _ in range(RESAMPLES))
    return [nearest_rank(deltas, 0.025), nearest_rank(deltas, 0.975)]


def compare(left, right, seed=SEED):
    rng = random.Random(seed)
    strata = {}
    for width in sorted(set(left["spans_ns"]) | set(right["spans_ns"])):
        a, b = left["spans_ns"].get(width, []), right["spans_ns"].get(width, [])
        row = {"hook": stats(a), "parent": stats(b)}
        if a and b:
            for name, q in (("median", 0.5), ("p95", 0.95)):
                row[f"{name}_delta_ns"] = nearest_rank(a, q) - nearest_rank(b, q)
                row[f"{name}_delta_ci95_ns"] = bootstrap(a, b, q, rng)
        strata[f"logical{width}"] = row
    kinds = sorted(set(left["records_by_kind"]) | set(right["records_by_kind"]))
    return {"by_logical_width": strata, "bootstrap": {"resamples": RESAMPLES, "seed": seed, "unit": "forward"},
            "joint_epochs": {"hook": left["joint_epochs"], "parent": right["joint_epochs"]},
            "incomplete_epochs": {"hook": left["incomplete_epochs"], "parent": right["incomplete_epochs"]},
            "records_by_kind_delta": {k: left["records_by_kind"].get(k, 0) - right["records_by_kind"].get(k, 0)
                                      for k in kinds},
            "records_delta_per_joint_epoch": (sum(left["records_by_kind"].values()) -
                                              sum(right["records_by_kind"].values())) / left["joint_epochs"]
            if left["joint_epochs"] else None}


def forward_split(windows):
    """T1 only: per logical width, the reducer's forward and settle durations (first commit to last return)."""
    joint = windows["validate_trace"].get("joint_ordinary_decode")
    require(type(joint) is dict, "windows report has no joint_ordinary_decode section")
    parts = {}
    for forward in joint["forwards"]:
        row = parts.setdefault(f"logical{forward['logical_rows']}", {"forward": [], "settle": []})
        for name in ("forward", "settle"):
            span = forward[name]
            if span["first_submit_ns"] is not None and span["last_checked_return_ns"] is not None:
                row[name].append(span["last_checked_return_ns"] - span["first_submit_ns"])
    return {width: {name: stats(values) for name, values in row.items()} for width, row in sorted(parts.items())}


def makespan(path):
    value = json.loads(Path(path).read_text())["own_output_four_task_makespan_seconds"]
    require(type(value) in (int, float) and value > 0, f"{path}: makespan seconds required")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left_trace")
    parser.add_argument("right_trace")
    parser.add_argument("--left-windows")
    parser.add_argument("--left-summary")
    parser.add_argument("--right-summary")
    args = parser.parse_args(argv)
    try:
        require((args.left_summary is None) == (args.right_summary is None), "both client summaries or neither")
        left, right = scan(Jsonl(args.left_trace)), scan(Jsonl(args.right_trace))
        result = {"schema": "hf2q.q4-trace.overhead.v1", **compare(left, right),
                  "trace_bytes": {"hook": Path(args.left_trace).stat().st_size,
                                  "parent": Path(args.right_trace).stat().st_size}}
        if args.left_windows:
            result["hook_forward_settle"] = forward_split(json.loads(Path(args.left_windows).read_text()))
        if args.left_summary:
            hook, parent = makespan(args.left_summary), makespan(args.right_summary)
            result["makespan_s"] = {"hook": hook, "parent": parent, "delta": hook - parent,
                                    "note": "one run per arm, descriptive"}
    except (KeyError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
