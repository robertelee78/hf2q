#!/usr/bin/env python3
"""Aggregate ADR-065 E3 shadow-draft records from one or more server logs."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "[ADR065_SHADOW_DRAFT_JSON] "
INPUT_SCHEMA = "hf2q.adr065.shadow-draft.v1"
OUTPUT_SCHEMA = "hf2q.adr065.shadow-acceptance.v1"
PROPOSERS = {"history", "mtp_k3"}
REQUIRED = {
    "schema",
    "slot",
    "proposer",
    "turn_type",
    "context_bucket",
    "start_position",
    "draft_count",
    "accepted_prefix",
    "proposal_cost_ns",
    "censored",
    "no_grammar",
    "no_repetition_penalty",
    "no_forced_thinking",
    "no_tool_trigger",
    "no_transforms",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_records(paths):
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            try:
                record = json.loads(line[marker + len(PREFIX) :])
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid shadow JSON: {error}") from error
            require(type(record) is dict and set(record) == REQUIRED,
                    f"{path}:{line_number}: shadow record keys do not match the closed schema")
            require(record["schema"] == INPUT_SCHEMA,
                    f"{path}:{line_number}: unsupported schema {record['schema']!r}")
            require(type(record["slot"]) is int and record["slot"] >= 0,
                    f"{path}:{line_number}: invalid slot")
            require(record["proposer"] in PROPOSERS,
                    f"{path}:{line_number}: invalid proposer")
            require(type(record["draft_count"]) is int and 1 <= record["draft_count"] <= 3,
                    f"{path}:{line_number}: invalid draft_count")
            require(type(record["accepted_prefix"]) is int
                    and 0 <= record["accepted_prefix"] <= record["draft_count"],
                    f"{path}:{line_number}: invalid accepted_prefix")
            require(type(record["proposal_cost_ns"]) is int and record["proposal_cost_ns"] >= 0,
                    f"{path}:{line_number}: invalid proposal cost")
            for key in ("censored", "no_grammar", "no_repetition_penalty",
                        "no_forced_thinking", "no_tool_trigger", "no_transforms"):
                require(type(record[key]) is bool, f"{path}:{line_number}: {key} is not boolean")
            expected_clear = all(record[key] for key in (
                "no_grammar", "no_repetition_penalty", "no_forced_thinking", "no_tool_trigger"))
            require(record["no_transforms"] == expected_clear,
                    f"{path}:{line_number}: no_transforms disagrees with component flags")
            records.append(record)
    require(records, "no ADR-065 shadow-draft records found")
    return records


def histogram(records):
    counts = Counter(record["accepted_prefix"] for record in records)
    return {str(prefix): counts.get(prefix, 0) for prefix in range(4)}


def grouped_histograms(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: histogram(group) for name, group in sorted(groups.items())}


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(records, sources):
    complete = [record for record in records if not record["censored"]]
    censored = len(records) - len(complete)
    require(complete, "all ADR-065 shadow-draft records are censored")
    proposers = {}
    for proposer in sorted(PROPOSERS):
        rows = [record for record in complete if record["proposer"] == proposer]
        costs = [record["proposal_cost_ns"] for record in rows]
        proposers[proposer] = {
            "records": len(rows),
            "accepted_prefix_histogram": histogram(rows),
            "by_slot": grouped_histograms(rows, "slot"),
            "by_turn_type": grouped_histograms(rows, "turn_type"),
            "by_context_bucket": grouped_histograms(rows, "context_bucket"),
            "no_transform_share": (sum(record["no_transforms"] for record in rows) / len(rows))
            if rows else None,
            "proposal_cost_ns": {
                "mean": statistics.fmean(costs) if costs else None,
                "median": statistics.median(costs) if costs else None,
                "p95": percentile(costs, 0.95) if costs else None,
            },
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "sources": [str(path) for path in sources],
        "records": len(records),
        "complete_records": len(complete),
        "censored_records": censored,
        "proposers": proposers,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = summarize(read_records(args.logs), args.logs)
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
