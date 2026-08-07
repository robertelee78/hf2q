#!/usr/bin/env python3
"""
ADR-015 iter8c-prep aggregator: per-submission GPU duration distribution
from xctrace "Metal System Trace" .trace bundles.

Why this exists:
  iter8b reported 14-37x per-kernel ratio gaps using HF2Q_MLX_KERNEL_PROFILE=1
  (242 sessions/token vs 1 in production). Side-by-side audit showed kernels
  are byte-equivalent to llama.cpp's, so the ratios are kprofile-mode
  artifacts, not production gaps. This aggregator parses production-mode
  metal-system-trace .trace bundles to capture per-dispatch GPU times
  (gpu-submission-id pairs function=1 start with function=2 end) for
  distribution comparison hf2q vs llama-cli. These intervals are keyed by
  GPU submission id; they are not individual Metal kernel dispatches.

Schema (from metal-gpu-execution-points):
  Each row has: timestamp, channel-id, function (1=start/2=end),
  slot-id, gpu-submission-id, accelerator-id, note.
  Pair start/end rows by gpu-submission-id; duration = end - start.

XML uses id/ref dictionary semantics (like aggregate_decode.py): a value
is defined once with `id="N"` and `fmt="..."`, then reused via `ref="N"`.

Output: per-binary distribution (count, sum, p50/p90/p95/p99, max),
plus a side-by-side comparison if multiple traces are passed. Sanitized
schema XML exported by xctrace can be supplied instead of a raw trace bundle.
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from statistics import median


XCTRACE = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"


def export_table(trace_path: str, schema: str = "metal-gpu-execution-points") -> str:
    """Run xctrace export to dump the named schema as XML."""
    if not os.path.isdir(trace_path):
        raise FileNotFoundError(f"trace bundle not found: {trace_path}")
    xpath = f'/trace-toc/run/data/table[@schema="{schema}"]'
    proc = subprocess.run(
        [XCTRACE, "export", "--input", trace_path, "--xpath", xpath],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def resolve_value(elem: ET.Element, dict_table: dict) -> str:
    """Resolve id/ref dictionary references inline.

    Each typed element either has an `id` attribute (definition; value
    in `fmt` attribute or text content) or a `ref` attribute (reference
    to a prior id).
    """
    rid = elem.get("id")
    ref = elem.get("ref")
    if rid is not None:
        # Definition site — store and return.
        val = elem.get("fmt") or (elem.text or "")
        dict_table[rid] = val
        return val
    if ref is not None:
        return dict_table.get(ref, "")
    # Inline literal.
    return elem.get("fmt") or (elem.text or "")


def parse_dispatches(xml_text: str) -> list:
    """Parse metal-gpu-execution-points XML into list of dispatch records.

    Returns: list of {sub_id, channel_id, slot_id, fn, t_ns} dicts.
    """
    root = ET.fromstring(xml_text)
    dispatches = []
    dict_table: dict = {}

    # Walk every <row> in the schema's table.
    for row in root.iter("row"):
        # Schema column order:
        #   1: start-time (ns) — <event-time>/<start-time>
        #   2: metal-command-buffer-id (channel-id)
        #   3: uint32 (function: 1=start, 2=end)
        #   4: uint32 (slot-id)
        #   5: metal-command-buffer-id (gpu-submission-id)
        #   6: uint64 (accelerator-id)
        #   7: string (note) or <sentinel/>
        # We resolve each in document order (id-defs come before refs).
        children = list(row)
        if len(children) < 5:
            continue

        # Column 1: start-time
        t_str = resolve_value(children[0], dict_table)
        # The text content is the integer ns; fmt is the formatted display.
        # Prefer text if numeric.
        t_text = (children[0].text or "").strip()
        try:
            t_ns = int(t_text) if t_text else int(t_str)
        except ValueError:
            try:
                t_ns = int(t_str)
            except ValueError:
                continue

        # Column 2: channel-id (CB lane)
        chan_str = resolve_value(children[1], dict_table)

        # Column 3: function (1=start, 2=end)
        fn_str = resolve_value(children[2], dict_table)
        try:
            fn = int(fn_str)
        except ValueError:
            continue

        # Column 4: slot-id
        slot_str = resolve_value(children[3], dict_table)

        # Column 5: gpu-submission-id (per-dispatch unique key)
        sub_str = resolve_value(children[4], dict_table)

        dispatches.append({
            "t_ns": t_ns,
            "channel": chan_str,
            "fn": fn,
            "slot": slot_str,
            "sub_id": sub_str,
        })

    return dispatches


def parse_submission_metadata(xml_text: str) -> dict:
    """Map each GPU submission id to its process and encoder metadata.

    The execution-points table has timing but no process column. The separate
    submission-to-command-buffer table supplies that ownership. A submission
    can appear more than once in that table, so retain a set rather than
    silently accepting the last row.
    """
    root = ET.fromstring(xml_text)
    dict_table: dict = {}
    metadata = defaultdict(lambda: {
        "processes": set(),
        "command_buffers": set(),
        "encoders": set(),
        "submission_types": set(),
        "models": set(),
    })

    for row in root.iter("row"):
        children = list(row)
        if len(children) < 13:
            continue
        # Schema columns: 2 = command-buffer-id, 3 = gpu-submission-id,
        # 6 = encoder-id, 12 = pid, 13 = process.
        command_buffer_id = resolve_value(children[1], dict_table)
        submission_id = resolve_value(children[2], dict_table)
        encoder_id = resolve_value(children[5], dict_table)
        submission_type = resolve_value(children[8], dict_table)
        resolve_value(children[11], dict_table)
        process = resolve_value(children[12], dict_table)
        model = resolve_value(children[13], dict_table) if len(children) > 13 else ""
        if submission_id and process:
            entry = metadata[submission_id]
            entry["processes"].add(process)
            if command_buffer_id:
                entry["command_buffers"].add(command_buffer_id)
            if encoder_id:
                entry["encoders"].add(encoder_id)
            if submission_type:
                entry["submission_types"].add(submission_type)
            if model:
                entry["models"].add(model)

    return metadata


def filter_process(paired: list, metadata: dict, process_substring: str) -> tuple:
    """Keep only intervals owned by a process whose display name matches."""
    kept = []
    unmatched = 0
    observed = Counter()
    for item in paired:
        entry = metadata.get(item["sub_id"])
        processes = entry["processes"] if entry else set()
        if not processes:
            unmatched += 1
            continue
        for process in processes:
            observed[process] += 1
        if any(process_substring in process for process in processes):
            kept.append({
                **item,
                "command_buffers": entry["command_buffers"],
                "encoders": entry["encoders"],
                "submission_types": entry["submission_types"],
                "models": entry["models"],
            })
    return kept, unmatched, observed


def parse_application_encoders(xml_text: str) -> list:
    """Parse application-created encoder intervals and ownership."""
    root = ET.fromstring(xml_text)
    dict_table: dict = {}
    encoders = []
    for row in root.iter("row"):
        children = list(row)
        if len(children) < 13:
            continue
        start_value = resolve_value(children[0], dict_table)
        duration_value = resolve_value(children[1], dict_table)
        # The first row defines the process inside the nested thread element;
        # later process columns reference that nested id.
        for nested in children[2].iter():
            resolve_value(nested, dict_table)
        process = resolve_value(children[3], dict_table)
        event_type = resolve_value(children[10], dict_table)
        command_buffer_id = resolve_value(children[11], dict_table)
        encoder_id = resolve_value(children[12], dict_table)
        try:
            start_ns = int((children[0].text or "").strip() or start_value)
            duration_ns = int((children[1].text or "").strip() or duration_value)
        except ValueError:
            continue
        encoders.append({
            "start_ns": start_ns,
            "end_ns": start_ns + duration_ns,
            "duration_ns": duration_ns,
            "process": process,
            "event_type": event_type,
            "command_buffer_id": command_buffer_id,
            "encoder_id": encoder_id,
        })
    return encoders


def pair_dispatches(rows: list) -> list:
    """Match function=1 (start) with function=2 (end) by gpu-submission-id.

    Returns: list of (sub_id, channel, start_ns, end_ns, duration_ns).
    """
    starts = {}
    paired = []
    unpaired_ends = 0

    for r in rows:
        key = r["sub_id"]
        if r["fn"] == 1:
            starts[key] = r
        elif r["fn"] == 2:
            s = starts.pop(key, None)
            if s is None:
                unpaired_ends += 1
                continue
            paired.append({
                "sub_id": key,
                "channel": s["channel"],
                "start_ns": s["t_ns"],
                "end_ns": r["t_ns"],
                "duration_ns": r["t_ns"] - s["t_ns"],
            })

    return paired, unpaired_ends, len(starts)  # leftover starts


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def merge_intervals(paired: list) -> list:
    intervals = sorted((item["start_ns"], item["end_ns"]) for item in paired)
    merged = []
    for start_ns, end_ns in intervals:
        if not merged or start_ns > merged[-1][1]:
            merged.append([start_ns, end_ns])
        else:
            merged[-1][1] = max(merged[-1][1], end_ns)
    return merged


def summarize(label: str, paired: list, span_start_ns=None, span_end_ns=None) -> dict:
    durs = [p["duration_ns"] for p in paired]
    if not durs:
        return {"label": label, "count": 0}
    merged = merge_intervals(paired)
    if span_start_ns is None:
        span_start_ns = merged[0][0]
    if span_end_ns is None:
        span_end_ns = merged[-1][1]
    busy_ns = sum(end_ns - start_ns for start_ns, end_ns in merged)
    span_ns = max(0, span_end_ns - span_start_ns)
    gaps = []
    gap_records = []
    cursor_ns = span_start_ns
    for start_ns, end_ns in merged:
        if start_ns > cursor_ns:
            gap_ns = start_ns - cursor_ns
            gaps.append(gap_ns)
            gap_records.append({"offset_ns": cursor_ns - span_start_ns,
                                "duration_ns": gap_ns})
        cursor_ns = max(cursor_ns, end_ns)
    if cursor_ns < span_end_ns:
        gap_ns = span_end_ns - cursor_ns
        gaps.append(gap_ns)
        gap_records.append({"offset_ns": cursor_ns - span_start_ns,
                            "duration_ns": gap_ns})
    submission_sum_ns = sum(durs)
    command_buffers = set().union(*(
        item.get("command_buffers", set()) for item in paired
    ))
    encoders = set().union(*(
        item.get("encoders", set()) for item in paired
    ))
    submission_types = Counter(
        value
        for item in paired
        for value in item.get("submission_types", set())
    )
    models = Counter(
        value
        for item in paired
        for value in item.get("models", set())
    )
    return {
        "label": label,
        "count": len(durs),
        "command_buffer_count": len(command_buffers),
        "encoder_count": len(encoders),
        "submission_types": dict(submission_types),
        "models": dict(models),
        "sum_ns": submission_sum_ns,
        "busy_ns": busy_ns,
        "overlap_ns": max(0, submission_sum_ns - busy_ns),
        "span_ns": span_ns,
        "idle_ns": max(0, span_ns - busy_ns),
        "utilization": (busy_ns / span_ns) if span_ns else 0.0,
        "gap_count": len(gaps),
        "gap_p50_ns": int(median(gaps)) if gaps else 0,
        "gap_p95_ns": int(percentile(gaps, 95)) if gaps else 0,
        "gap_max_ns": max(gaps) if gaps else 0,
        "gaps_over_1ms": sum(gap > 1_000_000 for gap in gaps),
        "gaps_over_5ms": sum(gap > 5_000_000 for gap in gaps),
        "gaps_over_10ms": sum(gap > 10_000_000 for gap in gaps),
        "top_gaps": sorted(gap_records, key=lambda gap: gap["duration_ns"],
                           reverse=True)[:10],
        "min_ns": min(durs),
        "p50_ns": int(median(durs)),
        "p90_ns": int(percentile(durs, 90)),
        "p95_ns": int(percentile(durs, 95)),
        "p99_ns": int(percentile(durs, 99)),
        "max_ns": max(durs),
        "mean_ns": int(sum(durs) / len(durs)),
    }


def fmt_us(ns: int) -> str:
    return f"{ns / 1000.0:.1f}"


def print_summary(s: dict):
    if s["count"] == 0:
        print(f"  {s['label']}: NO DISPATCHES PAIRED")
        return
    print(f"  {s['label']}:")
    print(f"    count    : {s['count']:>8d}")
    print(f"    cmd bufs : {s['command_buffer_count']:>8d}")
    print(f"    encoders : {s['encoder_count']:>8d}")
    print(f"    types    : {s['submission_types']}")
    print(f"    models   : {s['models']}")
    print(f"    sum      : {fmt_us(s['sum_ns']):>8s} µs submission time ({s['sum_ns']/1e9:.3f} s)")
    print(f"    busy     : {fmt_us(s['busy_ns']):>8s} µs union ({s['busy_ns']/1e9:.3f} s)")
    print(f"    overlap  : {fmt_us(s['overlap_ns']):>8s} µs")
    print(f"    span     : {fmt_us(s['span_ns']):>8s} µs ({s['span_ns']/1e9:.3f} s)")
    print(f"    idle     : {fmt_us(s['idle_ns']):>8s} µs ({s['idle_ns']/1e9:.3f} s)")
    print(f"    GPU busy : {s['utilization'] * 100:>7.2f}%")
    print(f"    gaps     : {s['gap_count']:>8d} (p50={fmt_us(s['gap_p50_ns'])} µs, p95={fmt_us(s['gap_p95_ns'])} µs, max={fmt_us(s['gap_max_ns'])} µs)")
    print(f"    gap tail : >1ms={s['gaps_over_1ms']} >5ms={s['gaps_over_5ms']} >10ms={s['gaps_over_10ms']}")
    print("    top gaps : " + ", ".join(
        f"@{gap['offset_ns'] / 1e6:.1f}ms={gap['duration_ns'] / 1e6:.1f}ms"
        for gap in s["top_gaps"]
    ))
    print(f"    mean     : {fmt_us(s['mean_ns']):>8s} µs/submission")
    print(f"    min      : {fmt_us(s['min_ns']):>8s} µs")
    print(f"    p50      : {fmt_us(s['p50_ns']):>8s} µs")
    print(f"    p90      : {fmt_us(s['p90_ns']):>8s} µs")
    print(f"    p95      : {fmt_us(s['p95_ns']):>8s} µs")
    print(f"    p99      : {fmt_us(s['p99_ns']):>8s} µs")
    print(f"    max      : {fmt_us(s['max_ns']):>8s} µs")


def print_comparison(a: dict, b: dict):
    print()
    print("Comparison:")
    print(f"  {'metric':<10s}  {a['label']:>20s}  {b['label']:>20s}  ratio")
    print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*5}")
    if a["count"] == 0 or b["count"] == 0:
        print("  (one side empty — comparison skipped)")
        return
    for key, label in [
        ("count", "count"),
        ("sum_ns", "sum"),
        ("busy_ns", "busy"),
        ("idle_ns", "idle"),
        ("mean_ns", "mean µs"),
        ("p50_ns", "p50 µs"),
        ("p90_ns", "p90 µs"),
        ("p99_ns", "p99 µs"),
        ("max_ns", "max µs"),
    ]:
        av, bv = a[key], b[key]
        ratio = (av / bv) if bv else float("inf")
        if key == "count":
            print(f"  {label:<10s}  {av:>20d}  {bv:>20d}  {ratio:>5.2f}×")
        else:
            print(f"  {label:<10s}  {fmt_us(av):>20s}  {fmt_us(bv):>20s}  {ratio:>5.2f}×")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", action="append",
                        help="Path to .trace bundle (repeatable; first 2 used for comparison)")
    source.add_argument("--xml", action="append",
                        help="Sanitized schema XML export (repeatable; first 2 used for comparison)")
    ap.add_argument("--mapping-xml", action="append", default=[],
                    help="Matching submission-to-command-buffer XML used to filter ownership")
    ap.add_argument("--encoder-xml", action="append", default=[],
                    help="Matching application-encoders XML for host encoding attribution")
    ap.add_argument("--app-encoder-limit", action="append", default=[], type=int,
                    help="Restrict each input to the first N application encoders")
    ap.add_argument("--process", action="append", default=[],
                    help="Process-name substring for each input; requires --mapping-xml")
    ap.add_argument("--label", action="append", default=[],
                    help="Label for each input (defaults to input basename)")
    ap.add_argument("--schema", default="metal-gpu-execution-points",
                    help="xctrace schema to query (default: metal-gpu-execution-points)")
    ap.add_argument("--window-start-ms", type=float, default=None,
                    help="Keep intervals starting at or after this offset from the first GPU interval")
    ap.add_argument("--window-end-ms", type=float, default=None,
                    help="Keep intervals ending at or before this offset from the first GPU interval")
    args = ap.parse_args()

    if args.window_start_ms is not None and args.window_start_ms < 0:
        ap.error("--window-start-ms must be non-negative")
    if args.window_end_ms is not None and args.window_end_ms < 0:
        ap.error("--window-end-ms must be non-negative")
    if (args.window_start_ms is not None and args.window_end_ms is not None
            and args.window_start_ms >= args.window_end_ms):
        ap.error("--window-start-ms must be less than --window-end-ms")
    if args.process and not args.mapping_xml:
        ap.error("--process requires --mapping-xml")
    if any(limit <= 0 for limit in args.app_encoder_limit):
        ap.error("--app-encoder-limit values must be positive")

    summaries = []
    inputs = args.trace if args.trace is not None else args.xml
    for i, t in enumerate(inputs):
        label = args.label[i] if i < len(args.label) else os.path.basename(t).replace(".trace", "")
        print(f"=== {label} ({t}) ===")
        if args.xml is not None:
            with open(t, encoding="utf-8") as xml_file:
                xml_text = xml_file.read()
        else:
            xml_text = export_table(t, args.schema)
        rows = parse_dispatches(xml_text)
        paired, unpaired_ends, leftover_starts = pair_dispatches(rows)
        if i < len(args.mapping_xml):
            with open(args.mapping_xml[i], encoding="utf-8") as mapping_file:
                metadata = parse_submission_metadata(mapping_file.read())
            process_substring = args.process[i] if i < len(args.process) else label
            paired, unmatched, observed = filter_process(
                paired, metadata, process_substring)
            print(f"  process filter  : {process_substring!r}")
            print(f"  unmapped pairs  : {unmatched}")
            print("  observed owners : " + ", ".join(
                f"{process}={count}" for process, count in observed.most_common(8)
            ))
        phase_encoders = None
        if i < len(args.encoder_xml):
            with open(args.encoder_xml[i], encoding="utf-8") as encoder_file:
                encoder_rows = parse_application_encoders(encoder_file.read())
            process_substring = args.process[i] if i < len(args.process) else label
            process_encoders = sorted(
                (row for row in encoder_rows if process_substring in row["process"]),
                key=lambda row: row["start_ns"],
            )
            if i < len(args.app_encoder_limit):
                phase_encoders = process_encoders[:args.app_encoder_limit[i]]
                phase_command_buffers = {
                    row["command_buffer_id"] for row in phase_encoders
                }
                paired = [
                    item for item in paired
                    if item.get("command_buffers", set()) & phase_command_buffers
                ]
                print(f"  encoder limit   : {args.app_encoder_limit[i]}")
        span_start_ns = None
        span_end_ns = None
        if paired and (args.window_start_ms is not None or args.window_end_ms is not None):
            origin_ns = min(item["start_ns"] for item in paired)
            span_start_ns = origin_ns + int((args.window_start_ms or 0.0) * 1_000_000)
            span_end_ns = (origin_ns + int(args.window_end_ms * 1_000_000)
                           if args.window_end_ms is not None
                           else max(item["end_ns"] for item in paired))
            clipped = []
            for item in paired:
                start_ns = max(item["start_ns"], span_start_ns)
                end_ns = min(item["end_ns"], span_end_ns)
                if start_ns < end_ns:
                    clipped.append({**item, "start_ns": start_ns,
                                    "end_ns": end_ns,
                                    "duration_ns": end_ns - start_ns})
            paired = clipped
        if i < len(args.encoder_xml):
            if phase_encoders is None:
                phase_encoders = [
                    row for row in process_encoders
                    if (span_start_ns is None or row["start_ns"] >= span_start_ns)
                    and (span_end_ns is None or row["start_ns"] < span_end_ns)
                ]
            encoder_duration_ns = sum(row["duration_ns"] for row in phase_encoders)
            print(f"  app encoders    : {len(phase_encoders)}")
            print(f"  app encode time : {encoder_duration_ns / 1e6:.3f} ms")
            print("  app events      : " + str(dict(Counter(
                row["event_type"] for row in phase_encoders
            ))))
        print(f"  rows parsed     : {len(rows)}")
        print(f"  submissions paired: {len(paired)}")
        print(f"  unpaired ends   : {unpaired_ends}")
        print(f"  leftover starts : {leftover_starts}")
        s = summarize(label, paired, span_start_ns, span_end_ns)
        print_summary(s)
        summaries.append(s)
        print()

    if len(summaries) >= 2:
        print_comparison(summaries[0], summaries[1])


if __name__ == "__main__":
    main()
