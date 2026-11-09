#!/usr/bin/env python3
"""Extract ADR-065 E4 critical-path composition from retained Q5 runs."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PREFIX = "[ADR065_CRITICAL_PATH_JSON] "
INPUT_SCHEMA = "hf2q.adr065.critical-path-interval.v1"
OUTPUT_SCHEMA = "hf2q.adr065.critical-path-report.v1"
CLOCK_ID = "CLOCK_MONOTONIC"
ELIGIBILITY = {"eligible", "blocked_transform", "no_draft", "unknown"}
T975 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448,
    15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def read_intervals(path):
    records = []
    for line_number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        where = f"{path}:{line_number}"
        try:
            record = json.loads(line[marker + len(PREFIX):])
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: invalid critical-path JSON: {error}") from error
        require(type(record) is dict, f"{where}: record is not an object")
        common = {"schema", "clock_id", "kind", "begin_ns", "end_ns"}
        require(record.get("schema") == INPUT_SCHEMA, f"{where}: unsupported schema")
        require(record.get("clock_id") == CLOCK_ID, f"{where}: unsupported clock")
        require(integer(record.get("begin_ns")) and integer(record.get("end_ns"))
                and 0 <= record["begin_ns"] < record["end_ns"],
                f"{where}: invalid interval bounds")
        if record.get("kind") == "decode":
            required = common | {"mode", "width", "lanes"}
            require(set(record) == required, f"{where}: decode keys do not match the closed schema")
            require(record["mode"] in {"joint", "solo"}, f"{where}: invalid decode mode")
            require(integer(record["width"]) and record["width"] >= 1,
                    f"{where}: invalid width")
            require(type(record["lanes"]) is list and len(record["lanes"]) == record["width"],
                    f"{where}: width disagrees with lanes")
            require(record["mode"] != "solo" or record["width"] == 1,
                    f"{where}: solo decode width is not one")
            slots = []
            for lane in record["lanes"]:
                require(type(lane) is dict and set(lane) == {"slot", "verify_eligibility"},
                        f"{where}: lane keys do not match the closed schema")
                require(integer(lane["slot"]) and lane["slot"] >= 0,
                        f"{where}: invalid lane slot")
                require(lane["verify_eligibility"] in ELIGIBILITY,
                        f"{where}: invalid verify eligibility")
                slots.append(lane["slot"])
            require(len(set(slots)) == len(slots), f"{where}: duplicate lane slot")
        elif record.get("kind") == "prefill":
            required = common | {"slot", "prompt_tokens", "chunk_start", "chunk_end"}
            require(set(record) == required, f"{where}: prefill keys do not match the closed schema")
            for key in ("slot", "prompt_tokens", "chunk_start", "chunk_end"):
                require(integer(record[key]) and record[key] >= 0, f"{where}: invalid {key}")
            require(record["chunk_start"] < record["chunk_end"] <= record["prompt_tokens"],
                    f"{where}: invalid prefill chunk")
        else:
            raise ValueError(f"{where}: invalid interval kind")
        records.append(record)
    require(records, f"{path}: no ADR-065 critical-path records found")
    records.sort(key=lambda record: (record["begin_ns"], record["end_ns"]))
    for left, right in zip(records, records[1:]):
        require(left["end_ns"] <= right["begin_ns"],
                f"{path}: engine intervals overlap ({left['kind']} and {right['kind']})")
    return records


def load_run(run_dir):
    receipt_path = run_dir / "receipt.json"
    summary_path = run_dir / "client" / "summary.json"
    log_path = run_dir / "server.stderr"
    for path in (receipt_path, summary_path, log_path):
        require(path.is_file(), f"{run_dir}: missing {path.name}")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("verdict", {}).get("verdict") in {"CalibrationPass", "AdmittedRunPass"},
            f"{receipt_path}: Q5 verdict did not pass")
    require(receipt.get("cleanup", {}).get("proven") is True,
            f"{receipt_path}: cleanup was not proven")
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema") == "hf2q.qwen-n4-coordinator-result.v1",
            f"{summary_path}: unsupported coordinator schema")
    require(summary.get("status") == "FourAgentProtocolCompleteAwaitingIndependentReview",
            f"{summary_path}: four-agent protocol did not complete")
    require(summary.get("f2_clock_id") == CLOCK_ID, f"{summary_path}: unsupported clock")
    start, end = summary.get("makespan_started_ns"), summary.get("makespan_finished_ns")
    require(integer(start) and integer(end) and 0 <= start < end,
            f"{summary_path}: invalid makespan")
    agents = summary.get("agents")
    require(type(agents) is dict and len(agents) == 4, f"{summary_path}: expected four agents")
    return receipt_path, summary_path, log_path, summary, read_intervals(log_path)


def client_intervals(summary, summary_path):
    start, end = summary["makespan_started_ns"], summary["makespan_finished_ns"]
    requests, gaps = [], []
    for agent_name, agent in sorted(summary["agents"].items()):
        require(agent.get("f2_clock_id") == CLOCK_ID,
                f"{summary_path}: {agent_name} uses an unsupported clock")
        turns = agent.get("turns")
        require(type(turns) is list and turns, f"{summary_path}: {agent_name} has no turns")
        previous_terminal = None
        first_send = None
        for index, turn in enumerate(turns):
            send = turn.get("send_monotonic_ns")
            terminal = turn.get("validated_terminal_monotonic_ns")
            require(integer(send) and integer(terminal) and start <= send < terminal <= end,
                    f"{summary_path}: {agent_name} turn {index} has invalid bounds")
            require(turn.get("f2_clock_id") == CLOCK_ID,
                    f"{summary_path}: {agent_name} turn {index} uses an unsupported clock")
            usage = turn.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            prompt_tokens, cached_tokens = usage.get("prompt_tokens"), details.get("cached_tokens")
            require(integer(prompt_tokens) and prompt_tokens > 0,
                    f"{summary_path}: {agent_name} turn {index} lacks prompt tokens")
            require(integer(cached_tokens) and 0 <= cached_tokens < prompt_tokens,
                    f"{summary_path}: {agent_name} turn {index} has invalid cached tokens")
            if previous_terminal is not None:
                require(previous_terminal <= send,
                        f"{summary_path}: {agent_name} turns overlap")
                if previous_terminal < send:
                    gaps.append({"kind": "tool_gap", "agent": agent_name,
                                 "begin_ns": previous_terminal, "end_ns": send})
            else:
                first_send = send
            requests.append({"agent": agent_name, "turn": turn.get("turn"),
                             "begin_ns": send, "end_ns": terminal,
                             "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens})
            previous_terminal = terminal
        completed = agent.get("task_completed_monotonic_ns")
        require(integer(completed) and previous_terminal <= completed <= end,
                f"{summary_path}: {agent_name} has invalid completion time")
        if start < first_send:
            gaps.append({"kind": "idle", "agent": agent_name,
                         "begin_ns": start, "end_ns": first_send})
        if previous_terminal < completed:
            gaps.append({"kind": "tool_gap", "agent": agent_name,
                         "begin_ns": previous_terminal, "end_ns": completed})
        if completed < end:
            gaps.append({"kind": "idle", "agent": agent_name,
                         "begin_ns": completed, "end_ns": end})
    return requests, gaps


def queue_intervals(requests, engine, source):
    first_prefills = [record for record in engine
                      if record["kind"] == "prefill"]
    used = set()
    queues = []
    for request in sorted(requests, key=lambda row: row["begin_ns"]):
        candidates = [
            (index, span) for index, span in enumerate(first_prefills)
            if index not in used
            and span["prompt_tokens"] == request["prompt_tokens"]
            and span["chunk_start"] == request["cached_tokens"]
            and request["begin_ns"] <= span["begin_ns"] < request["end_ns"]
        ]
        require(len(candidates) == 1,
                f"{source}: request {request['agent']} turn {request['turn']} matched "
                f"{len(candidates)} initial prefill slices")
        index, span = candidates[0]
        used.add(index)
        queues.append({"kind": "queue", "agent": request["agent"],
                       "turn": request["turn"], "slot": span["slot"],
                       "begin_ns": request["begin_ns"], "end_ns": span["begin_ns"]})
    return queues


def covered(interval, point):
    return interval["begin_ns"] <= point < interval["end_ns"]


def critical_partition(start, end, engine, queues, gaps):
    boundaries = {start, end}
    for interval in engine + queues + gaps:
        if interval["end_ns"] > start and interval["begin_ns"] < end:
            boundaries.add(max(start, interval["begin_ns"]))
            boundaries.add(min(end, interval["end_ns"]))
    ordered = sorted(boundaries)
    totals = Counter()
    for left, right in zip(ordered, ordered[1:]):
        if left == right:
            continue
        point = left
        active_engine = [span for span in engine if covered(span, point)]
        require(len(active_engine) <= 1, "engine intervals overlap in critical partition")
        if active_engine:
            span = active_engine[0]
            category = span["mode"] if span["kind"] == "decode" else "prefill"
        elif any(covered(interval, point) for interval in queues):
            category = "queue"
        elif any(covered(interval, point) and interval["kind"] == "tool_gap" for interval in gaps):
            category = "tool_gap"
        else:
            category = "idle_or_unattributed"
        totals[category] += right - left
    for key in ("joint", "solo", "prefill", "queue", "tool_gap", "idle_or_unattributed"):
        totals[key] += 0
    require(sum(totals.values()) == end - start, "critical partition does not equal makespan")
    return dict(totals)


def summarize_run(run_dir):
    receipt_path, summary_path, log_path, summary, engine = load_run(run_dir)
    start, end = summary["makespan_started_ns"], summary["makespan_finished_ns"]
    require(all(start <= span["begin_ns"] < span["end_ns"] <= end for span in engine),
            f"{log_path}: engine interval lies outside the baseline makespan")
    requests, gaps = client_intervals(summary, summary_path)
    queues = queue_intervals(requests, engine, log_path)
    partition = critical_partition(start, end, engine, queues, gaps)
    joint = [span for span in engine if span["kind"] == "decode" and span["mode"] == "joint"]
    widths = Counter(span["width"] for span in joint)
    width_time = Counter()
    eligible_weighted_ns = 0.0
    eligible_lane_ns = 0
    total_lane_ns = 0
    for span in joint:
        duration = span["end_ns"] - span["begin_ns"]
        eligible = sum(lane["verify_eligibility"] == "eligible" for lane in span["lanes"])
        width_time[span["width"]] += duration
        eligible_weighted_ns += duration * eligible / span["width"]
        eligible_lane_ns += duration * eligible
        total_lane_ns += duration * span["width"]
    makespan_ns = end - start
    return {
        "run": run_dir.name,
        "sources": {"receipt": str(receipt_path), "summary": str(summary_path),
                    "server_log": str(log_path)},
        "q5_verdict": json.loads(receipt_path.read_text())["verdict"]["verdict"],
        "makespan": {"begin_ns": start, "end_ns": end, "duration_ns": makespan_ns},
        "interval_counts": {"engine": len(engine), "requests": len(requests),
                            "queue": len(queues), "tool_or_idle": len(gaps)},
        "queue_intervals": sorted(
            queues, key=lambda interval: (interval["begin_ns"], interval["agent"])
        ),
        "client_gap_intervals": sorted(
            gaps, key=lambda interval: (interval["begin_ns"], interval["agent"])
        ),
        "critical_path_ns": partition,
        "joint_width_distribution": {
            str(width): {"turns": widths[width], "duration_ns": width_time[width]}
            for width in sorted(widths)
        },
        "verify_eligibility": {
            "estimand": "sum(joint_turn_duration * eligible_lanes / width) / baseline_makespan",
            "eligible_lane_weighted_ns": eligible_weighted_ns,
            "eligible_lane_ns": eligible_lane_ns,
            "total_lane_ns": total_lane_ns,
            "lane_time_share": eligible_lane_ns / total_lane_ns if total_lane_ns else None,
            "f": eligible_weighted_ns / makespan_ns,
        },
    }


def clustered_interval(values):
    require(values, "no run clusters")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"runs": 1, "mean": mean, "ci95": None, "method": "unavailable for one run"}
    require(len(values) - 1 in T975, "run count exceeds the Student-t table")
    half = T975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    return {"runs": len(values), "mean": mean,
            "ci95": [max(0.0, mean - half), min(1.0, mean + half)],
            "method": "two-sided 95% Student-t interval over independent run-level f clusters"}


def summarize(run_dirs):
    runs = [summarize_run(path) for path in run_dirs]
    return {
        "schema": OUTPUT_SCHEMA,
        "runs": runs,
        "run_clustered_f": clustered_interval([
            run["verify_eligibility"]["f"] for run in runs
        ]),
        "independent_review_required": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path,
                        help="retained Q5 run directories containing receipt.json, server.stderr and client/")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    body = json.dumps(summarize(args.runs), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
