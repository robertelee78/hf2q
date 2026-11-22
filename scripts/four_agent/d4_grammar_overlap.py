#!/usr/bin/env python3
"""Strict, SHA-bound diagnostic of DFlash2 grammar/GPU interval intersections.

Times are native mach-clock seconds. kernel_start/end describe CPU driver
scheduling, not GPU execution. This analyzer neither authenticates the input
nor establishes registration, semantic parity, qualification or speedup.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

WORK_SCHEMA = "hf2q.dflash2.grammar_work.v1"
REPORT_SCHEMA = "hf2q.dflash2.grammar_overlap.v1"
HEAD = "qwen35.joint_decode.output_head"
LAYERS = "qwen35.joint_decode.layers"
SCOPES = {
    "head": frozenset({HEAD}),
    "target-verifier-body": frozenset({"qwen35.embedding.gpu_token_ids", LAYERS, HEAD}),
}
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_COMMANDS_PER_RECORD = 4096
MAX_MARKS_PER_RECORD = 65_536
MAX_COMMANDS = 2_000_000
MAX_MARKS = 4_000_000
MAX_U64 = 2**64 - 1
OUTCOMES = ("invalid", "prepared", "planning_consumed", "divergence_discarded")
LANE_MARK = re.compile(r"grammar\.slot([0-3])\.simulate_(begin|end)\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, maximum=MAX_U64):
    return type(value) is int and 0 <= value <= maximum


def seconds(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def union(intervals):
    """Merge touching/overlapping intervals, following f2_gpu_timing.union_ns."""
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        elif end > result[-1][1]:
            result[-1][1] = end
    return result


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection_seconds(left, right):
    left, right = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        total += max(0.0, end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _marks(record):
    marks = record.get("marks")
    require(type(marks) is list and 2 <= len(marks) <= MAX_MARKS_PER_RECORD,
            "bounded nonempty marks array")
    named = {}
    previous = 0
    for index, mark in enumerate(marks):
        require(type(mark) is list and len(mark) == 2 and type(mark[0]) is str
                and 0 < len(mark[0]) <= 256 and seconds(mark[1], positive=True),
                "typed finite native-clock mark")
        name, at = mark
        require(at >= previous, "marks are not in native-clock order")
        previous = at
        named.setdefault(name, []).append((index, at))

    def one(name, optional=False):
        entries = named.get(name, [])
        require(len(entries) == 1 or (optional and not entries), f"one {name} mark")
        return entries[0] if entries else None

    begin, end = one("begin"), one("end")
    verify_begin, verify_end = one("verify_begin"), one("verify_end")
    completed = one("joint.head_completed")
    require(begin[0] == 0 and end[0] == len(marks) - 1
            and begin[0] < verify_begin[0] < completed[0] < verify_end[0] < end[0]
            and begin[1] < end[1] and verify_begin[1] < verify_end[1],
            "finished cycle/verify/head marker ordering")
    host_begin, host_end = one("joint.host_work_begin", True), one("joint.host_work_end", True)
    grammar_begin, grammar_end = one("grammar.simulate_begin", True), one("grammar.simulate_end", True)
    submitted = one("joint.head_submitted", True)
    sync = one("joint.head_profile_sync_no_overlap", True)
    require((host_begin is None) == (host_end is None)
            and (grammar_begin is None) == (grammar_end is None)
            and (host_begin is None) == (grammar_begin is None),
            "complete paired host/grammar callback marks")
    if host_begin is not None:
        require((submitted is None) != (sync is None), "one async submission or explicit sync marker")
        route = submitted if submitted is not None else sync
        order = [verify_begin, route, host_begin, grammar_begin, grammar_end, host_end, completed, verify_end]
        require(all(a[0] < b[0] for a, b in zip(order, order[1:])),
                "nested callback and head completion ordering")
    else:
        require(submitted is None and sync is None, "head route marker without callback")
    return marks, {
        "begin": begin[1], "end": end[1], "verify_begin": verify_begin[1],
        "verify_end": verify_end[1], "head_completed": completed[1],
        "host": [] if host_begin is None else [(host_begin[1], host_end[1])],
        "grammar": [] if grammar_begin is None else [(grammar_begin[1], grammar_end[1])],
        "route_at": None if host_begin is None else route[1], "sync": sync is not None,
    }


def _commands(record, clock, scope, previous):
    commands = record.get("commands")
    require(type(commands) is list and 0 < len(commands) <= MAX_COMMANDS_PER_RECORD,
            "bounded nonempty commands array")
    selected, heads = [], []
    layers = 0
    for command in commands:
        require(type(command) is dict, "command object")
        label = command.get("label")
        require(label is None or (type(label) is str and len(label) <= 256), "command label")
        require(command.get("completed") is True and type(command.get("waited_in_drain")) is bool,
                "completed command with typed drain flag")
        for name in ("encode_begin", "commit_begin", "commit_end", "wait_begin", "wait_end",
                     "kernel_start", "kernel_end", "gpu_start", "gpu_end"):
            require(seconds(command.get(name)), f"finite nonnegative command {name}")
        encode, commit, committed = (command[k] for k in ("encode_begin", "commit_begin", "commit_end"))
        gpu = command["gpu_start"], command["gpu_end"]
        require(0 <= encode <= commit <= committed and commit > 0
                and 0 < gpu[0] < gpu[1] and commit <= gpu[0], "ordered positive command GPU/commit times")
        wait = command["wait_begin"], command["wait_end"]
        require(wait == (0, 0) or (commit <= wait[0] <= wait[1] and gpu[1] <= wait[1]),
                "paired ordered command wait times")
        require(command["kernel_start"] <= command["kernel_end"], "ordered CPU driver-scheduling times")
        require(commit >= previous[0], "command commit ordering across capture")
        counters = command.get("dispatches_at_commit"), command.get("barriers_at_commit")
        require(all(integer(value) for value in counters) and counters[0] >= previous[1]
                and counters[1] >= previous[2], "typed nondecreasing command counters")
        previous = commit, *counters
        owned = clock["verify_begin"] <= encode <= commit <= committed <= clock["verify_end"]
        if owned and label in SCOPES["target-verifier-body"]:
            require(gpu[1] <= clock["head_completed"], "target GPU completion after checked head mark")
            if label == HEAD:
                heads.append(command)
            if label == LAYERS:
                layers += 1
            if label in SCOPES[scope]:
                selected.append(gpu)
    require(heads, "no target output-head command encoded and committed inside verify")
    if scope == "target-verifier-body":
        require(layers > 0, "body scope requires target layer commands, not a silently head-only scope")
    if clock["route_at"] is not None:
        require(all(c["commit_end"] <= clock["route_at"] for c in heads), "head route precedes head commit")
    if clock["sync"]:
        require(all(c["gpu_end"] <= clock["route_at"] for c in heads), "sync marker precedes head GPU completion")
    return selected, previous


def _proposal_spans(marks):
    lanes, proposals = {}, {}
    lane = proposal = grammar = None
    grammar_seen = False
    for name, at in marks:
        match = LANE_MARK.fullmatch(name)
        if name == "grammar.simulate_begin":
            require(grammar is None and not grammar_seen and lane is None and proposal is None,
                    "one outer grammar extent")
            grammar, grammar_seen = at, True
        elif name == "grammar.simulate_end":
            require(grammar is not None and lane is None and proposal is None, "unclosed grammar lane/proposal")
            grammar = None
        elif match:
            slot, edge = int(match[1]), match[2]
            require(grammar is not None, "lane outside grammar callback")
            if edge == "begin":
                require(lane is None and proposal is None and slot not in lanes, "unique nonnested physical lane")
                lane = slot, at
                proposals[slot] = []
            else:
                require(lane is not None and lane[0] == slot and proposal is None, "matching closed lane")
                lanes[slot] = lane[1], at
                lane = None
        elif name == "grammar.proposal_begin":
            require(lane is not None and proposal is None, "proposal must begin in its own lane")
            proposal = at
        elif name == "grammar.proposal_end":
            require(lane is not None and proposal is not None, "matching lane-contained proposal end")
            proposals[lane[0]].append((proposal, at))
            require(len(proposals[lane[0]]) <= 7, "at most seven proposal spans per lane")
            proposal = None
        elif name.startswith("grammar."):
            raise ValueError(f"unknown grammar marker: {name}")
    require(grammar is None and lane is None and proposal is None, "unclosed grammar proposal extent")
    return lanes, proposals


def _work(record, marks, clock, gpu):
    lanes, proposals = _proposal_spans(marks)
    present = "grammar_work" in record or "grammar_work_schema" in record
    if not present:
        require(not lanes, "proposal/lane marks without work metadata")
        return {"metadata_status": "missing", "useful_work_eligible": False,
                "ineligibility_reasons": ["missing_grammar_work_metadata"]}
    require(record.get("grammar_work_schema") == WORK_SCHEMA, "exact grammar work schema")
    work = record.get("grammar_work")
    require(type(work) is list and len(work) == 4, "four physical grammar work slots")
    observed = {slot for slot, value in enumerate(work) if value is not None}
    require(observed == set(lanes), "work slots must match observed lane spans exactly")
    require(len(observed) <= record["active_lanes"], "observed work exceeds active lanes")
    require(bool(observed) == bool(clock["grammar"]), "work slots must match grammar callback presence")
    groups = {state: {outcome: [] for outcome in OUTCOMES} for state in ("active", "awaiting_trigger")}
    actual_proposals = attempts = 0
    lane_rows = []
    for slot in sorted(observed):
        value = work[slot]
        require(type(value) is dict and set(value) == {"actual_proposals", "awaiting_trigger_at_start", "attempts"},
                "closed grammar lane metadata")
        count = value["actual_proposals"]
        require(integer(count, 7) and type(value["awaiting_trigger_at_start"]) is bool,
                "typed actual proposals and initial awaiting state")
        entries = value["attempts"]
        require(type(entries) is list and len(entries) == 7, "seven attempt entries")
        attempted = []
        tail = False
        for entry in entries:
            if entry is None:
                tail = True
            else:
                require(not tail and type(entry) is dict
                        and set(entry) == {"awaiting_trigger", "eos", "outcome"}
                        and type(entry["awaiting_trigger"]) is bool and type(entry["eos"]) is bool
                        and type(entry["outcome"]) is str and entry["outcome"] in OUTCOMES,
                        "closed contiguous typed attempt prefix")
                attempted.append(entry)
        require(len(attempted) <= count and len(attempted) == len(proposals[slot]),
                "attempts must join exact proposal spans in array order")
        require(count == 0 or attempted, "nonempty proposals require an attempted prefix")
        if len(attempted) < count:
            require(attempted[-1]["eos"] or attempted[-1]["outcome"] == "invalid", "unexplained unattempted tail")
        previous_awaiting = value["awaiting_trigger_at_start"]
        unused = None
        for index, (entry, span) in enumerate(zip(attempted, proposals[slot])):
            awaiting, outcome = entry["awaiting_trigger"], entry["outcome"]
            require((index != 0 or awaiting == previous_awaiting)
                    and (previous_awaiting or not awaiting), "invalid awaiting-trigger progression")
            previous_awaiting = awaiting
            if outcome == "invalid" or entry["eos"]:
                require(index == len(attempted) - 1, "invalid/EOS attempt must stop preparation")
            if outcome == "planning_consumed":
                require(unused is None, "consumed transitions must be a matching prefix")
            elif outcome != "invalid":
                require(unused is None or unused == outcome, "prepared/discarded suffix cannot switch disposition")
                unused = outcome
            groups["awaiting_trigger" if awaiting else "active"][outcome].append(span)
        actual_proposals += count
        attempts += len(attempted)
        lane_rows.append({"slot": slot, "actual_proposals": count, "attempts": len(attempted),
                          "awaiting_trigger_at_start": value["awaiting_trigger_at_start"]})
    require(actual_proposals <= record["selected_rows"] - record["active_lanes"],
            "actual proposals exceed selected rows after lane anchors")
    measured = {}
    for state, outcomes in groups.items():
        measured[state] = {
            outcome: {"attempts": len(spans), "span_seconds": duration(spans),
                      "gpu_overlap_seconds": intersection_seconds(spans, gpu)}
            for outcome, spans in outcomes.items()
        }
    consumed = measured["active"]["planning_consumed"]
    reasons = []
    if clock["sync"]:
        reasons.append("explicit_synchronous_head_profile")
    if consumed["attempts"] == 0:
        reasons.append("no_active_planning_consumed_transitions")
    return {"metadata_status": "observed", "actual_proposals": actual_proposals,
            "attempts": attempts, "lanes": lane_rows, "by_state_and_outcome": measured,
            "useful_work_eligible": not reasons, "ineligibility_reasons": reasons,
            "active_consumed_overlap_seconds": consumed["gpu_overlap_seconds"],
            "active_consumed_overlap_observed": not reasons and consumed["gpu_overlap_seconds"] > 0}


def _cycle(record, scope, previous):
    require(type(record) is dict and record.get("finished") is True, "finished cycle object required")
    require(integer(record.get("active_lanes"), 4) and record["active_lanes"] > 0
            and integer(record.get("selected_rows"), 32) and record["selected_rows"] >= record["active_lanes"]
            and record["selected_rows"] <= 8 * record["active_lanes"]
            and integer(record.get("accepted_rows"), record["selected_rows"])
            and type(record.get("gpu_acceptance")) is bool, "typed bounded cycle summary")
    require(integer(record.get("dropped_commands")) and record["dropped_commands"] == 0
            and integer(record.get("dropped_marks")) and record["dropped_marks"] == 0,
            "typed zero dropped commands/marks required")
    marks, clock = _marks(record)
    require(clock["begin"] >= previous[3], "cycle clock ordering across capture")
    gpu, counters = _commands(record, clock, scope, previous[:3])
    result = {
        "interval_capture_valid": True, "synchronous_head_profile": clock["sync"],
        "selected_gpu_commands": len(gpu), "excluded_commands": len(record["commands"]) - len(gpu),
        "gpu_union_seconds": duration(gpu),
        "grammar_callback_seconds": duration(clock["grammar"]),
        "grammar_callback_overlap_seconds": intersection_seconds(clock["grammar"], gpu),
        "host_callback_overlap_seconds": intersection_seconds(clock["host"], gpu),
        "grammar_work": _work(record, marks, clock, gpu),
    }
    return result, (*counters, clock["end"])


def analyze(path, *, expected_sha256, scope):
    """Analyze original JSONL bytes; reject the entire capture on any bad record.

    Both keyword arguments are mandatory. ValueError means malformed, unbounded
    or mismatched evidence; OSError means the input could not be read.
    """
    require(type(expected_sha256) is str and re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256),
            "expected SHA256 must be 64 hexadecimal characters")
    require(type(scope) is str and scope in SCOPES, "explicit head or target-verifier-body scope required")
    digest = hashlib.sha256()
    rows = []
    byte_count = command_count = mark_count = 0
    previous = (0, 0, 0, 0)
    with Path(path).open("rb") as source:
        while True:
            raw = source.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            require(len(raw) <= MAX_LINE_BYTES, "original-byte line limit exceeded")
            byte_count += len(raw)
            require(byte_count <= MAX_FILE_BYTES, "original-byte file limit exceeded")
            require(len(rows) < MAX_RECORDS, "record limit exceeded")
            digest.update(raw)
            number = len(rows) + 1
            try:
                record = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_object, parse_constant=_constant)
                row, previous = _cycle(record, scope, previous)
                command_count += len(record["commands"])
                mark_count += len(record["marks"])
                require(command_count <= MAX_COMMANDS and mark_count <= MAX_MARKS, "capture command/mark limit exceeded")
            except (ValueError, OverflowError, RecursionError) as error:
                raise ValueError(f"line {number}: {error}") from error
            rows.append(row)
    require(rows, "empty timeline")
    require(digest.hexdigest() == expected_sha256.lower(), "original-byte SHA256 mismatch")
    eligible = sum(row["grammar_work"]["useful_work_eligible"] for row in rows)
    positive = sum(row["grammar_work"].get("active_consumed_overlap_observed", False) for row in rows)
    return {
        "schema": REPORT_SCHEMA, "scope": scope, "target_labels": sorted(SCOPES[scope]),
        "input": {"sha256": digest.hexdigest(), "expected_sha256_matched": True,
                  "original_bytes": byte_count, "records": len(rows),
                  "commands": command_count, "marks": mark_count},
        "interval_capture_valid": True,
        "summary": {"cycles": len(rows), "useful_work_eligible_cycles": eligible,
                    "active_consumed_overlap_cycles": positive,
                    "legacy_missing_metadata_cycles": sum(row["grammar_work"]["metadata_status"] == "missing" for row in rows),
                    "synchronous_head_profile_cycles": sum(row["synchronous_head_profile"] for row in rows),
                    "grammar_callback_overlap_seconds": sum(row["grammar_callback_overlap_seconds"] for row in rows),
                    "active_consumed_overlap_seconds": sum(row["grammar_work"].get("active_consumed_overlap_seconds", 0) for row in rows)},
        "claims": {"input_authenticity_proven": False, "binary_model_config_source_binding_proven": False,
                   "registration_proven": False, "ordinary_path_runtime_semantic_parity_proven": False,
                   "release_family_runtime_quality_performance_qualification": False,
                   "speedup_proven": False, "rewrites_rejected_registered_t2_head_verdict": False},
        "cycles": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--scope", required=True, choices=tuple(SCOPES))
    args = parser.parse_args(argv)
    try:
        report = analyze(args.timeline, expected_sha256=args.expected_sha256, scope=args.scope)
    except (OSError, ValueError, OverflowError, RecursionError) as error:
        print(f"d4_grammar_overlap: {error}", file=sys.stderr)
        return 1
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
