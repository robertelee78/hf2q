#!/usr/bin/env python3
"""Reduce an HF2Q_DFLASH2_CYCLE_TIMELINE file.

Each input line is one DFlash2 verify cycle: the scheduler's host marks and
every command buffer committed since the previous line, with host encode,
commit and wait times and Metal's driver-scheduling (kernelStartTime,
kernelEndTime) and GPU-execution (GPUStartTime, GPUEndTime) intervals, all in
seconds of mach absolute time.

For the cycles of one lane count (default 4) this prints medians of:

- cycle wall, the union of GPU execution inside the cycle, and GPU idle;
- where the GPU idles: before the first command starts, between commands
  (split into "the next command was not committed yet" and "committed but not
  started"), and after the last command ends;
- host time by scheduler stage and what remains outside every command's
  encode span;
- per command, in commit order: dispatches, barriers, host encode span,
  commit call, commit-to-GPU-start (split into queued behind the previous
  command and start latency), GPU time, and any host wait with its wake-up
  latency after the GPU finished;
- the gap between consecutive cycles.

usage: dflash2_cycle_timeline.py FILE [--lanes N] [--skip K] [--json OUT]
                                 [--dump INDEX]
"""

import argparse
import json
import statistics
import sys


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else float("nan")


def ms(seconds):
    return seconds * 1e3


def reduce_cycle(cycle, previous_dispatches, previous_barriers):
    marks = dict(cycle["marks"])
    begin, end = marks["begin"], marks["end"]
    commands = []
    dispatches, barriers = previous_dispatches, previous_barriers
    for command in cycle["commands"]:
        own_dispatches = command["dispatches_at_commit"] - dispatches
        own_barriers = command["barriers_at_commit"] - barriers
        dispatches = command["dispatches_at_commit"]
        barriers = command["barriers_at_commit"]
        if command["commit_begin"] < begin:
            continue  # committed before this cycle (prefill, an aborted cycle)
        if not command["completed"]:
            return None, dispatches, barriers
        commands.append(dict(command, dispatches=own_dispatches, barriers=own_barriers))
    if not commands:
        return None, dispatches, barriers

    # GPU execution union inside the cycle, and the idle gaps around it.
    intervals = sorted((c["gpu_start"], c["gpu_end"]) for c in commands)
    busy = 0.0
    cursor = begin
    for start, stop in intervals:
        start, stop = max(start, cursor), min(stop, end)
        if stop > start:
            busy += stop - start
            cursor = stop
    wall = end - begin

    idle_before_first = max(0.0, commands[0]["gpu_start"] - begin)
    idle_after_last = max(0.0, end - max(c["gpu_end"] for c in commands))
    idle_uncommitted = 0.0  # GPU had nothing: the next command was still being encoded
    idle_latency = 0.0  # next command committed, not yet started
    rows = []
    previous_gpu_end = None
    for command in commands:
        encode = (
            command["commit_begin"] - command["encode_begin"]
            if command["encode_begin"] > 0.0
            else 0.0
        )
        ready = command["commit_end"]
        if previous_gpu_end is None:
            queued = 0.0
            latency = command["gpu_start"] - ready
        else:
            queued = max(0.0, previous_gpu_end - ready)
            latency = command["gpu_start"] - max(ready, previous_gpu_end)
            gap = command["gpu_start"] - previous_gpu_end
            if gap > 0.0:
                starved = min(gap, max(0.0, ready - previous_gpu_end))
                idle_uncommitted += starved
                idle_latency += gap - starved
        waited = command["wait_begin"] > 0.0
        rows.append(
            {
                "label": command["label"] or "(unlabeled)",
                "dispatches": command["dispatches"],
                "barriers": command["barriers"],
                "encode_begin_ms": ms(command["encode_begin"] - begin)
                if command["encode_begin"] > 0.0
                else None,
                "commit_ms": ms(command["commit_end"] - begin),
                "encode_span_ms": ms(encode),
                "commit_call_ms": ms(command["commit_end"] - command["commit_begin"]),
                "kernel_start_ms": ms(command["kernel_start"] - begin),
                "kernel_sched_ms": ms(command["kernel_end"] - command["kernel_start"]),
                "gpu_start_ms": ms(command["gpu_start"] - begin),
                "gpu_end_ms": ms(command["gpu_end"] - begin),
                "commit_to_gpu_start_ms": ms(command["gpu_start"] - ready),
                "queued_ms": ms(queued),
                "start_latency_ms": ms(latency),
                "gpu_ms": ms(command["gpu_end"] - command["gpu_start"]),
                "wait_ms": ms(command["wait_end"] - command["wait_begin"]) if waited else None,
                "wake_ms": ms(command["wait_end"] - command["gpu_end"]) if waited else None,
            }
        )
        previous_gpu_end = max(previous_gpu_end or 0.0, command["gpu_end"])

    # Host marks in recording order: offset from the cycle's begin and the
    # time since the previous mark.
    mark_rows = []
    previous_mark = begin
    for name, seconds in cycle["marks"]:
        mark_rows.append({"name": name, "at_ms": ms(seconds - begin), "since_previous_ms": ms(seconds - previous_mark)})
        previous_mark = seconds
    # Residency-set commits (mlx-native marks each one it issues at a command
    # commit): how many, and the host time inside them.
    residency_begin = None
    residency_commits = 0
    residency_seconds = 0.0
    for name, seconds in cycle["marks"]:
        if name == "residency_commit_begin":
            residency_begin = seconds
        elif name == "residency_commit_end" and residency_begin is not None:
            residency_commits += 1
            residency_seconds += seconds - residency_begin
            residency_begin = None

    encode_total = sum(row["encode_span_ms"] for row in rows)
    wait_total = sum(row["wait_ms"] or 0.0 for row in rows)
    summary = {
        "wall_ms": ms(wall),
        "gpu_busy_ms": ms(busy),
        "gpu_idle_ms": ms(wall - busy),
        "gpu_sum_ms": sum(row["gpu_ms"] for row in rows),
        "idle_before_first_ms": ms(idle_before_first),
        "idle_uncommitted_ms": ms(idle_uncommitted),
        "idle_start_latency_ms": ms(idle_latency),
        "idle_after_last_ms": ms(idle_after_last),
        "host_encode_spans_ms": encode_total,
        "host_waits_ms": wait_total,
        "host_other_ms": ms(wall) - encode_total - wait_total,
        "commands": len(rows),
        "dispatches": sum(row["dispatches"] for row in rows),
        "barriers": sum(row["barriers"] for row in rows),
        "host_waits": sum(1 for row in rows if row["wait_ms"] is not None),
        "residency_commits": residency_commits,
        "residency_commit_ms": ms(residency_seconds),
    }
    return {"summary": summary, "rows": rows, "marks": mark_rows, "begin": begin, "end": end}, dispatches, barriers


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--skip", type=int, default=3, help="leading cycles to drop")
    parser.add_argument("--json", help="write the medians as JSON")
    parser.add_argument("--dump", type=int, help="print one cycle (index among kept) in full")
    args = parser.parse_args()

    cycles = []
    dispatches = barriers = 0
    previous_end = None
    previous_lanes = None
    with open(args.file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cycle = json.loads(line)
            reduced, dispatches, barriers = reduce_cycle(cycle, dispatches, barriers)
            if reduced is None:
                previous_end = None
                continue
            foreign = any(
                command["commit_begin"] < reduced["begin"] for command in cycle["commands"]
            )
            reduced["lanes"] = cycle["active_lanes"]
            reduced["accepted_rows"] = cycle["accepted_rows"]
            reduced["gpu_acceptance"] = cycle["gpu_acceptance"]
            # The gap from the previous cycle counts only when nothing else
            # (a prefill, another lane count) ran in between.
            reduced["gap_ms"] = (
                ms(reduced["begin"] - previous_end)
                if previous_end is not None
                and not foreign
                and previous_lanes == cycle["active_lanes"]
                else None
            )
            previous_end, previous_lanes = reduced["end"], cycle["active_lanes"]
            cycles.append(reduced)

    kept = [cycle for cycle in cycles if cycle["lanes"] == args.lanes][args.skip :]
    if not kept:
        sys.exit(f"no {args.lanes}-lane cycles in {args.file}")
    # Most cycles share one command structure; reduce per-command medians
    # over the most common one so rows line up.
    shapes = {}
    for cycle in kept:
        shape = (
            tuple(row["label"] for row in cycle["rows"]),
            tuple(mark["name"] for mark in cycle["marks"]),
        )
        shapes.setdefault(shape, []).append(cycle)
    (shape, mark_shape), same = max(shapes.items(), key=lambda item: len(item[1]))

    summary = {
        key: median(cycle["summary"][key] for cycle in kept) for key in kept[0]["summary"]
    }
    summary["cycles"] = len(kept)
    summary["cycles_in_modal_command_shape"] = len(same)
    summary["gap_between_cycles_ms"] = median(cycle["gap_ms"] for cycle in kept)
    summary["gap_samples"] = sum(1 for cycle in kept if cycle["gap_ms"] is not None)
    summary["accepted_rows_mean"] = statistics.fmean(cycle["accepted_rows"] for cycle in kept)
    summary["gpu_acceptance_cycles"] = sum(1 for cycle in kept if cycle["gpu_acceptance"])

    print(f"{args.file}: {len(kept)} cycles of {args.lanes} lanes (first {args.skip} dropped)")
    for key, value in summary.items():
        print(f"  {key:32s} {value:10.3f}" if isinstance(value, float) else f"  {key:32s} {value:>10}")

    columns = [
        "dispatches",
        "barriers",
        "encode_begin_ms",
        "commit_ms",
        "encode_span_ms",
        "commit_call_ms",
        "kernel_sched_ms",
        "gpu_start_ms",
        "gpu_end_ms",
        "queued_ms",
        "start_latency_ms",
        "gpu_ms",
        "wait_ms",
        "wake_ms",
    ]
    table = []
    for index, label in enumerate(shape):
        row = {"index": index, "label": label}
        for column in columns:
            row[column] = median(cycle["rows"][index][column] for cycle in same)
        table.append(row)
    print(f"\nper command, median over {len(same)} cycles with the modal command shape")
    print("  #  " + " ".join(f"{column[:-3] if column.endswith('_ms') else column:>13s}" for column in columns) + "  label")
    for row in table:
        print(
            f"  {row['index']:<2d} "
            + " ".join(f"{row[column]:13.3f}" for column in columns)
            + f"  {row['label']}"
        )

    mark_table = [
        {
            "name": name,
            "at_ms": median(cycle["marks"][index]["at_ms"] for cycle in same),
            "since_previous_ms": median(cycle["marks"][index]["since_previous_ms"] for cycle in same),
        }
        for index, name in enumerate(mark_shape)
    ]
    print(f"\nhost marks, median over the same {len(same)} cycles")
    print(f"  {'at':>10s} {'since prev':>10s}  mark")
    for mark in mark_table:
        print(f"  {mark['at_ms']:10.3f} {mark['since_previous_ms']:10.3f}  {mark['name']}")

    if args.dump is not None:
        print(f"\ncycle {args.dump} in full")
        print(json.dumps(kept[args.dump], indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "commands": table, "marks": mark_table}, handle, indent=1)


if __name__ == "__main__":
    main()
