#!/usr/bin/env python3
"""
ADR-015 iter8c-prep aggregator: per-dispatch GPU duration distribution
from xctrace "Metal System Trace" .trace bundles.

Why this exists:
  iter8b reported 14-37x per-kernel ratio gaps using HF2Q_MLX_KERNEL_PROFILE=1
  (242 sessions/token vs 1 in production). Side-by-side audit showed kernels
  are byte-equivalent to llama.cpp's, so the ratios are kprofile-mode
  artifacts, not production gaps. This aggregator parses production-mode
  metal-system-trace .trace bundles to capture per-dispatch GPU times
  (gpu-submission-id pairs function=1 start with function=2 end) for
  distribution comparison hf2q vs llama-cli.

Schema (from metal-gpu-execution-points):
  Each row has: timestamp, channel-id, function (1=start/2=end),
  slot-id, gpu-submission-id, accelerator-id, note.
  Pair start/end rows by gpu-submission-id; duration = end - start.

XML uses id/ref dictionary semantics (like aggregate_decode.py): a value
is defined once with `id="N"` and `fmt="..."`, then reused via `ref="N"`.

Output: per-binary distribution (count, sum, p50/p90/p95/p99, max),
plus a side-by-side comparison if multiple traces are passed.
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
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


def summarize(label: str, paired: list) -> dict:
    durs = [p["duration_ns"] for p in paired]
    if not durs:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": len(durs),
        "sum_ns": sum(durs),
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
    print(f"    sum      : {fmt_us(s['sum_ns']):>8s} µs total ({s['sum_ns']/1e9:.3f} s)")
    print(f"    mean     : {fmt_us(s['mean_ns']):>8s} µs/dispatch")
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
    ap.add_argument("--trace", action="append", required=True,
                    help="Path to .trace bundle (repeatable; first 2 used for comparison)")
    ap.add_argument("--label", action="append", default=[],
                    help="Label for each --trace (defaults to bundle basename)")
    ap.add_argument("--schema", default="metal-gpu-execution-points",
                    help="xctrace schema to query (default: metal-gpu-execution-points)")
    args = ap.parse_args()

    summaries = []
    for i, t in enumerate(args.trace):
        label = args.label[i] if i < len(args.label) else os.path.basename(t).replace(".trace", "")
        print(f"=== {label} ({t}) ===")
        xml_text = export_table(t, args.schema)
        rows = parse_dispatches(xml_text)
        paired, unpaired_ends, leftover_starts = pair_dispatches(rows)
        print(f"  rows parsed     : {len(rows)}")
        print(f"  dispatches paired: {len(paired)}")
        print(f"  unpaired ends   : {unpaired_ends}")
        print(f"  leftover starts : {leftover_starts}")
        s = summarize(label, paired)
        print_summary(s)
        summaries.append(s)
        print()

    if len(summaries) >= 2:
        print_comparison(summaries[0], summaries[1])


if __name__ == "__main__":
    main()
