#!/usr/bin/env python3
"""ADR-063 Q4-TRACE (#663) validation: acceptance predicate, Q2 v2 isolation, exactness rule.

  acceptance WINDOWS_OR_REPORT   the bound predicate on a reduced traced run (q2_traced_freeze windows
                                 JSON, or a bare validate_trace report)
  isolation TRACE [--windows W] [--all-compositions]
                                 f2_trace.qualify_window on TRACE and on TRACE in its pre-hook shape
                                 (joint forward_input/forward_witness/forward_input_end dropped, joint Known
                                 scopes relabelled OpaqueAuxiliary, joint source removed from the header
                                 policy) must agree exactly for every checked window (design §5.1 P5, §5.2
                                 check 2). Windows: the frozen ones of W (q2_traced_freeze output), or every
                                 composition at those arrivals with --all-compositions.
  classify --t1-o1 J --g1 pass|fail --t0-o0 J --o1-o1d J [--repeat J]
                                 the pre-registered exactness rule (design §5.2) over
                                 `q1_decision_policy.py transcripts` outputs; rows top to bottom, first match

Read-only and CPU-only.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import f2_trace
from f2_native_trace import JOINT_OPERATION, JOINT_SOURCE

COMPOSITIONS = ((1, 0), (0, 1), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4))
# The queued-demand policy admits only the Qwen serial witness sources (f2_demand.validate_policy), so a
# trace whose policy names the joint source is refused there by design (§3.6): no window to compare.
POLICIES = ("ready_now_v1", "forward_demand_v2")
TRANSCRIPTS = "hf2q.q1.transcript-equality.v1"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def acceptance(document):
    """The bound predicate: no unmapped attempt, and every joint transaction witnessed (at least one)."""
    report = document.get("validate_trace", document)
    joint = report.get(JOINT_OPERATION)
    require(type(joint) is dict, "report has no joint_ordinary_decode section (policy lacks the joint source)")
    checks = {
        "unmapped_operator_attempts == 0": report["unmapped_operator_attempts"] == 0,
        "unmapped_by_operation.joint_ordinary_decode == 0": report["unmapped_by_operation"][JOINT_OPERATION] == 0,
        "joint.unwitnessed == 0": joint["unwitnessed"] == 0,
        "joint.witnessed == joint.transactions > 0": joint["witnessed"] == joint["transactions"] > 0,
    }
    return {"schema": "hf2q.q4-trace.acceptance.v1", "accepted": all(checks.values()),
            "failed": [name for name, ok in checks.items() if not ok],
            "transactions": joint["transactions"], "witnessed": joint["witnessed"],
            "unwitnessed": joint["unwitnessed"], "unmapped_operator_attempts": report["unmapped_operator_attempts"],
            "boundary_ambiguous_commands": joint["boundary_ambiguous_commands"],
            "pre_forward_commands": joint["pre_forward_commands"], "width_histogram": joint["width_histogram"]}


class Jsonl:
    """Re-iterable streamed JSONL trace: every pass re-reads the file; no record is retained."""

    def __init__(self, path):
        self.path = Path(path)

    def __iter__(self):
        with self.path.open("rb") as handle:
            for line in handle:
                yield json.loads(line)


class PreHook:
    """Re-iterable view of a trace in the shape it would have without the joint witness hook. A joint epoch
    holds exactly one Known scope, the joint witness's (inputs are epoch-unique), so it is found by epoch."""

    def __init__(self, records):
        require(f2_trace.records_source(records), "re-iterable trace records required")
        self.records = records
        self.epochs = {r["payload"]["epoch"] for r in records
                       if r["kind"] == "forward_input" and r["payload"]["source"] == JOINT_SOURCE}

    def __iter__(self):
        seq = 0
        for record in self.records:
            kind, p = record["kind"], record["payload"]
            if record["seq"] == 0:
                record = copy.deepcopy(record)
                policy = record["payload"]["native_observer"]["witness_manifest"]
                policy["input_sources"] = [s for s in policy["input_sources"] if s != JOINT_SOURCE]
            elif kind == "forward_input_end" or (kind in ("forward_input", "forward_witness") and
                                                 p["source"] == JOINT_SOURCE):
                continue
            elif kind == "scope_definition" and p["membership"] == "Known" and p["scope"]["epoch"] in self.epochs:
                record = dict(record, payload=dict(p, operation="opaque_auxiliary",
                                                   membership="OpaqueAuxiliary", rows=[]))
            yield dict(record, seq=seq)
            seq += 1


def isolation(records, windows):
    """qualify_window on the trace and on its pre-hook shape, for every (composition, arrival) x policy."""
    stripped = PreHook(records)
    require(stripped.epochs, "trace has no joint forward input: nothing to isolate")
    results, differing = [], []
    for composition, arrival in windows:
        for policy in POLICIES:
            hooked = f2_trace.qualify_window(records, composition, arrival, population_policy=policy)
            plain = f2_trace.qualify_window(stripped, composition, arrival, population_policy=policy)
            item = {"composition": list(composition), "arrival_ns": arrival, "population_policy": policy,
                    "status": hooked["status"], "identical": hooked == plain}
            results.append(item)
            if hooked != plain:
                differing.append(dict(item, hooked=hooked, pre_hook=plain))
    return {"schema": "hf2q.q4-trace.q2-isolation.v1", "joint_epochs": len(stripped.epochs),
            "windows": results, "identical": not differing, "differing": differing,
            "queued_demand": "not compared: the queued policy refuses the joint source (design §3.6)"}


def transcript_verdict(document, name):
    require(type(document) is dict and document.get("schema") == TRANSCRIPTS and document.get("runs") == 2 and
            document.get("classification") in ("ExactTranscripts", "DivergedTranscripts"),
            f"{name}: a two-run q1_decision_policy transcripts result required")
    return "Exact" if document["classification"] == "ExactTranscripts" else "Diverged"


def classify(t1_o1, g1, t0_o0, o1_o1d, repeat=None):
    """Design §5.2 pre-registered rule (amended before any run: O1' determinism control). Rows are evaluated
    top to bottom and the first match decides. Qwen3.8 N4 serving switches serial/joint routes with arrival
    timing (a Q1 Class B route), so a cross-run transcript divergence is hook evidence only once the untraced
    hook binary is shown run-to-run deterministic (O1 vs O1'); G1 is the fixed-schedule hook test."""
    require(g1 in ("pass", "fail"), "G1 outcome must be pass or fail")
    hook, observer = transcript_verdict(t1_o1, "T1 O1"), transcript_verdict(t0_o0, "T0 O0")
    control = transcript_verdict(o1_o1d, "O1 O1'")
    rows = {"T1 O1": hook, "G1": g1, "T0 O0": observer, "O1 O1'": control}
    if g1 == "fail":
        verdict, outcome = "HookAttributable", "Story fails: fix the hook (G1 proves output bytes changed)."
    elif hook == "Exact":
        verdict, outcome = "Exact", "Exactness criterion met."
    elif control == "Diverged":
        verdict, outcome = ("ScheduleNondeterministic", "Hook cleared by G1; untraced serving is not run-to-run "
                            "transcript-deterministic (O1 vs O1'). To the operator with all comparisons, divergent "
                            "turns, T1's joint/serial mix and the recommended ADR re-wording.")
    elif observer == "Diverged":
        verdict, outcome = ("ObserverAttributable", "Hook cleared; observer alone diverges. To the operator with "
                            "all comparisons, divergent turns and the recommended ADR re-wording.")
    elif repeat is None:
        verdict, outcome = "Unresolved", "Run one more pair T1' and O1'' with identical inputs, then re-classify."
    else:
        rows["T1' O1''"] = transcript_verdict(repeat, "T1' O1''")
        if rows["T1' O1''"] == "Exact":
            verdict, outcome = ("ScheduleNondeterministic", "To the operator, as ObserverAttributable, with all "
                                "comparisons and divergent turns.")
        else:
            verdict, outcome = ("HookAttributable", "Story fails: the repeat pair diverged again while the "
                                "untraced hook binary was run-to-run deterministic.")
    require(repeat is None or (hook, g1, control, observer) == ("Diverged", "pass", "Exact", "Exact"),
            "a repeat pair is allowed only for the Unresolved row")
    return {"schema": "hf2q.q4-trace.exactness.v2", "inputs": rows, "classification": verdict, "outcome": outcome}


def load(path):
    return json.loads(Path(path).read_text())


def frozen_windows(path, all_compositions):
    windows = load(path)
    require(windows.get("schema") == "hf2q.q2.traced-windows.v1", "q2_traced_freeze windows output required")
    frozen = [(tuple(w["composition"]), w["arrival_ns"]) for w in windows["windows"]]
    if not all_compositions:
        return frozen
    return [(c, arrival) for arrival in sorted({a for _, a in frozen}) for c in COMPOSITIONS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("acceptance")
    a.add_argument("report")
    i = sub.add_parser("isolation")
    i.add_argument("trace")
    i.add_argument("--windows", required=True, help="q2_traced_freeze windows-<run>.json of the same run")
    i.add_argument("--all-compositions", action="store_true")
    c = sub.add_parser("classify")
    c.add_argument("--t1-o1", required=True)
    c.add_argument("--g1", required=True, choices=("pass", "fail"))
    c.add_argument("--t0-o0", required=True)
    c.add_argument("--o1-o1d", required=True, help="O1 vs O1' (untraced hook binary run twice): determinism control")
    c.add_argument("--repeat", help="T1' vs O1'' (only for the Unresolved row)")
    args = parser.parse_args(argv)
    try:
        if args.command == "acceptance":
            result = acceptance(load(args.report))
        elif args.command == "isolation":
            result = isolation(Jsonl(args.trace), frozen_windows(args.windows, args.all_compositions))
        else:
            result = classify(load(args.t1_o1), args.g1, load(args.t0_o0), load(args.o1_o1d),
                              None if args.repeat is None else load(args.repeat))
    except (KeyError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.command == "acceptance":
        return 0 if result["accepted"] else 1
    if args.command == "isolation":
        return 0 if result["identical"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
