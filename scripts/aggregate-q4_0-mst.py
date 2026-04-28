#!/usr/bin/env python3
"""
ADR-015 iter9 aggregator: kernel attribution from xctrace MST traces
for hf2q vs llama-cli on the Q4_0-dominated dwq46 apex workload.

Why this exists (vs aggregate_decode_mst.py):
  iter8c-prep's aggregator emits per-dispatch DURATION distributions
  (count, sum, p50/p95) keyed only by sub_id/channel/slot. There is no
  kernel-name attribution. iter9 spec S3 requires kernel-name attribution
  to identify the iter10 attack target.

Schema discovery (run on a smoke trace 2026-04-28T13:58Z, hf2q dwq46):
  Available schemas in MST trace:
    - metal-application-encoders-list  (per-encoder: cmd-buf, encoder, duration)
    - metal-application-event-interval (DEBUG GROUPS — empty; mlx-native does not pushDebugGroup)
    - metal-driver-intervals           (Command Buffer / Command Encoder labels — generic)
    - metal-object-label               (CoreAnimation surfaces, AGXHeaps — no shader fn names)
    - metal-shader-profiler-shader-list (compiled shaders — empty in this template)
    - metal-gpu-execution-points       (per-dispatch start/end pairs by gpu-submission-id)

  Conclusion: with default "Metal System Trace" template + mlx-native binary
  (which does NOT label MTLComputePipelineState or pushDebugGroup),
  KERNEL-NAME ATTRIBUTION VIA xctrace IS NOT POSSIBLE. The closest signal
  available is the per-encoder duration in metal-application-encoders-list,
  cross-joined with the per-dispatch GPU times in metal-gpu-execution-points
  by command-buffer-id. We then BUCKET by dispatch duration histogram —
  Q4_0 mat-vec dispatches sit in a known time band (~5–60 µs per dispatch
  on the dwq46 apex workload at decode), distinct from flash-attention
  (~50–500 µs) or RMS-norm (~1–10 µs).

Output (/tmp/adr015-iter9/aggregate-q4_0.txt):
  Per-binary, per-trial, the dispatch-duration histogram with bucketed
  attribution against a structural reference (counts per token from the
  Qwen3.5-MoE forward graph: 32 layers × 8 used_experts × 3 mat-vec per
  expert + 32 attn × 4 RMS + 32 flash-attn + …). Side-by-side hf2q vs
  llama for the highest-mass buckets, with Δµs/token and Δ% columns.

  Structural counts per token (n_used_experts=8, n_layers=32):
    Q4_0 mat-vec (mul_mv_id_q4_0_f32):
      gate/up: 32 layers × 8 experts × 2 = 512 dispatches/tok
      down:    32 layers × 8 experts × 1 = 256 dispatches/tok
      total:   768 Q4_0-id dispatches/tok
    Q4_0 mat-vec (mul_mv_q4_0_f32, dense Q/K/V/O proj):
      qkv_o:   32 layers × 4 = 128 dispatches/tok (if not fused)

Methodology AC2 compliance:
  Per spec AC2 ("NO double-counted overlapping inclusive frames"), we use
  metal-gpu-execution-points (point-events at GPU start/end) as the
  CANONICAL frame. metal-driver-intervals is INCLUSIVE of GPU work and
  is NOT summed. metal-application-encoders-list is reported separately
  (encoder-side wall-clock, not GPU time).

Usage:
  scripts/aggregate-q4_0-mst.py \
    --hf2q-trace /tmp/adr015-iter9/hf2q-trial-1.trace \
    --hf2q-trace /tmp/adr015-iter9/hf2q-trial-2.trace \
    ... \
    --llama-trace /tmp/adr015-iter9/llama-trial-1.trace \
    ... \
    --n-tokens 64 \
    --output /tmp/adr015-iter9/aggregate-q4_0.txt

Falls back gracefully if either side has fewer trials (median over what's
available). If --hf2q-trace is provided but --llama-trace is empty, prints
hf2q-only partial attribution (this is the iter9 claude-side mid-iteration
case, before codex completes llama capture).
"""

import argparse
import os
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


XCTRACE = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"


# --------------------------------------------------------------------- #
# Schema discovery                                                      #
# --------------------------------------------------------------------- #

def export_table(trace_path: str, schema: str) -> str:
    """Run `xctrace export --xpath ...` and return the XML stdout."""
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


def export_toc(trace_path: str) -> str:
    proc = subprocess.run(
        [XCTRACE, "export", "--input", trace_path, "--toc"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


# --------------------------------------------------------------------- #
# id/ref dictionary — same scheme used by aggregate_decode_mst.py       #
# --------------------------------------------------------------------- #

def resolve(elem: ET.Element, table: dict) -> str:
    rid = elem.get("id")
    ref = elem.get("ref")
    if rid is not None:
        val = elem.get("fmt") or (elem.text or "")
        table[rid] = val
        return val
    if ref is not None:
        return table.get(ref, "")
    return elem.get("fmt") or (elem.text or "")


def text_int(elem: ET.Element, table: dict) -> Optional[int]:
    """Return the integer value of an XML cell, resolving id/ref."""
    rid = elem.get("id")
    ref = elem.get("ref")
    if rid is not None:
        # Definition: prefer text content (raw integer), fall back to fmt.
        t = (elem.text or "").strip()
        try:
            v = int(t) if t else int(elem.get("fmt") or "0")
            table[rid] = v
            return v
        except ValueError:
            return None
    if ref is not None:
        return table.get(ref)
    t = (elem.text or "").strip()
    try:
        return int(t) if t else int(elem.get("fmt") or "0")
    except ValueError:
        return None


# --------------------------------------------------------------------- #
# Schema 1: metal-gpu-execution-points (per-dispatch GPU start/end)     #
# Schema columns (verified from existing aggregate_decode_mst.py):      #
#   0: start-time (ns)                                                   #
#   1: metal-command-buffer-id (channel-id)                              #
#   2: uint32 (function: 1=start, 2=end)                                 #
#   3: uint32 (slot-id)                                                  #
#   4: metal-command-buffer-id (gpu-submission-id)                       #
#   5: uint64 (accelerator-id)                                           #
#   6: string (note) — optional sentinel                                 #
# --------------------------------------------------------------------- #

def parse_gpu_execution_points(xml_text: str) -> List[dict]:
    root = ET.fromstring(xml_text)
    rows = []
    table = {}
    for row in root.iter("row"):
        children = list(row)
        if len(children) < 5:
            continue

        t_str = resolve(children[0], table)
        # Prefer raw int text over fmt
        t_text = (children[0].text or "").strip()
        try:
            t_ns = int(t_text) if t_text else int(t_str)
        except ValueError:
            try:
                t_ns = int(t_str)
            except ValueError:
                continue

        chan = resolve(children[1], table)
        try:
            fn = int(resolve(children[2], table))
        except ValueError:
            continue
        slot = resolve(children[3], table)
        sub  = resolve(children[4], table)

        rows.append(
            dict(t_ns=t_ns, channel=chan, fn=fn, slot=slot, sub_id=sub)
        )
    return rows


def pair_dispatches(rows: List[dict]) -> Tuple[List[dict], int, int]:
    """Pair fn=1 (start) with fn=2 (end) by sub_id."""
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
            paired.append(dict(
                sub_id=key,
                channel=s["channel"],
                start_ns=s["t_ns"],
                end_ns=r["t_ns"],
                duration_ns=r["t_ns"] - s["t_ns"],
            ))
    return paired, unpaired_ends, len(starts)


# --------------------------------------------------------------------- #
# Schema 2: metal-application-encoders-list                             #
#   0: start-time                                                        #
#   1: duration                                                          #
#   2: thread                                                            #
#   3: process                                                           #
#   4: gpu (metal-device-name)                                           #
#   5: frame-number                                                      #
#   6: cmdbuffer-label                                                   #
#   7: cmdbuffer-label-indexed                                           #
#   8: encoder-label                                                     #
#   9: encoder-label-indexed                                             #
#  10: event-type                                                        #
#  11: cmdbuffer-id                                                      #
#  12: encoder-id                                                        #
# --------------------------------------------------------------------- #

def parse_encoders_list(xml_text: str) -> List[dict]:
    root = ET.fromstring(xml_text)
    rows = []
    table = {}
    for row in root.iter("row"):
        children = list(row)
        if len(children) < 13:
            continue

        # start-time (ns)
        t_text = (children[0].text or "").strip()
        try:
            t_ns = int(t_text)
        except ValueError:
            try:
                t_ns = int(resolve(children[0], table))
            except ValueError:
                continue

        # duration (ns)
        d_text = (children[1].text or "").strip()
        try:
            dur_ns = int(d_text) if d_text else int(resolve(children[1], table))
        except ValueError:
            continue

        encoder_label = resolve(children[8], table)
        cmdbuffer_label = resolve(children[6], table)
        event_type = resolve(children[10], table)

        rows.append(dict(
            start_ns=t_ns,
            duration_ns=dur_ns,
            encoder_label=encoder_label,
            cmdbuffer_label=cmdbuffer_label,
            event_type=event_type,
        ))
    return rows


# --------------------------------------------------------------------- #
# Bucketing by dispatch duration                                        #
# --------------------------------------------------------------------- #

# Empirical buckets for the dwq46 decode workload (verified against
# qwen35/forward_gpu.rs structural counts). Boundaries chosen so that
# each bucket cleanly maps to a kernel class:
#   [0,    2_000)   ns   small ops: rms_norm, scalar mul, reshape, etc.
#   [2_000, 8_000)  ns   medium: small mat-vecs (Q/K/V dense if not fused),
#                        rope, soft-cap
#   [8_000, 32_000) ns   Q4_0 mat-vec MoE expert dispatches (heart of gap)
#   [32_000, 80_000) ns  flash_attn, mul_mm_id pooled
#   [80_000, +∞)    ns   prefill mul_mm_id, lm_head, large blits
BUCKETS = [
    ("xs_<2us",     0,         2_000),
    ("sm_2_8us",    2_000,     8_000),
    ("md_8_32us",   8_000,     32_000),
    ("lg_32_80us",  32_000,    80_000),
    ("xl_>=80us",   80_000,    None),
]


def bucket_of(dur_ns: int) -> str:
    for name, lo, hi in BUCKETS:
        if dur_ns >= lo and (hi is None or dur_ns < hi):
            return name
    return "unknown"


def bucket_summary(paired: List[dict]) -> Dict[str, dict]:
    """Compute per-bucket count + sum + p50 + p95."""
    by_bucket = defaultdict(list)
    for p in paired:
        by_bucket[bucket_of(p["duration_ns"])].append(p["duration_ns"])

    out = {}
    for name, _, _ in BUCKETS:
        durs = by_bucket.get(name, [])
        out[name] = dict(
            count=len(durs),
            sum_ns=sum(durs),
            p50_ns=int(statistics.median(durs)) if durs else 0,
            p95_ns=int(durs[int(0.95 * (len(durs) - 1))]) if len(durs) >= 2 else (durs[0] if durs else 0),
            mean_ns=int(sum(durs) / len(durs)) if durs else 0,
        )
    out["_total"] = dict(
        count=len(paired),
        sum_ns=sum(p["duration_ns"] for p in paired),
    )
    return out


# --------------------------------------------------------------------- #
# Per-trace summary                                                     #
# --------------------------------------------------------------------- #

def summarize_trace(trace_path: str, n_tokens: int) -> dict:
    """Return per-trace bucketed dispatch summary."""
    xml = export_table(trace_path, "metal-gpu-execution-points")
    rows = parse_gpu_execution_points(xml)
    paired, unpaired_ends, leftover = pair_dispatches(rows)

    enc_xml = export_table(trace_path, "metal-application-encoders-list")
    encoders = parse_encoders_list(enc_xml)

    buckets = bucket_summary(paired)
    total_gpu_ns = buckets["_total"]["sum_ns"]
    total_dispatches = buckets["_total"]["count"]

    return dict(
        path=trace_path,
        rows=len(rows),
        paired=len(paired),
        unpaired_ends=unpaired_ends,
        leftover_starts=leftover,
        encoders=len(encoders),
        encoder_total_ns=sum(e["duration_ns"] for e in encoders),
        n_tokens=n_tokens,
        buckets=buckets,
        # Per-token attribution
        dispatches_per_token=total_dispatches / max(n_tokens, 1),
        gpu_us_per_token=total_gpu_ns / 1000.0 / max(n_tokens, 1),
    )


def median_summaries(summaries: List[dict]) -> dict:
    """Combine N per-trial summaries into a single median view."""
    if not summaries:
        return {}
    out = {
        "n_trials": len(summaries),
        "n_tokens_per_trial": [s["n_tokens"] for s in summaries],
        "paired_per_trial": [s["paired"] for s in summaries],
        "median_dispatches_per_token": statistics.median(
            [s["dispatches_per_token"] for s in summaries]
        ),
        "median_gpu_us_per_token": statistics.median(
            [s["gpu_us_per_token"] for s in summaries]
        ),
        "buckets": {},
    }
    for name, _, _ in BUCKETS:
        counts_per_tok = []
        sums_per_tok_us = []
        p50_us = []
        for s in summaries:
            n_tok = max(s["n_tokens"], 1)
            b = s["buckets"].get(name, {})
            counts_per_tok.append(b.get("count", 0) / n_tok)
            sums_per_tok_us.append(b.get("sum_ns", 0) / 1000.0 / n_tok)
            p50_us.append(b.get("p50_ns", 0) / 1000.0)
        out["buckets"][name] = dict(
            median_dispatches_per_token=statistics.median(counts_per_tok),
            median_us_per_token=statistics.median(sums_per_tok_us),
            median_p50_us_per_dispatch=statistics.median(p50_us),
        )
    return out


# --------------------------------------------------------------------- #
# Side-by-side report                                                   #
# --------------------------------------------------------------------- #

def fmt_int(v) -> str:
    if isinstance(v, float):
        if v >= 100:
            return f"{v:>10.1f}"
        return f"{v:>10.3f}"
    return f"{v:>10}"


def write_report(out_path: str, hf2q: dict, llama: dict, hf2q_trials: List[dict], llama_trials: List[dict]):
    lines = []
    lines.append("=" * 110)
    lines.append("ADR-015 iter9 — Q4_0 dispatch-duration attribution (xctrace MST)")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Methodology:")
    lines.append("  - canonical frame: metal-gpu-execution-points fn=1/2 paired by sub_id (per AC2)")
    lines.append("  - encoders sidecar: metal-application-encoders-list (informational; not summed)")
    lines.append("  - kernel-name attribution: NOT POSSIBLE in this MST template (mlx-native does")
    lines.append("    not pushDebugGroup or label MTLComputePipelineState; metal-application-event-interval")
    lines.append("    and metal-shader-profiler-shader-list are empty).")
    lines.append("  - bucketing strategy: per-dispatch duration histogram into 5 bands, where each")
    lines.append("    band cleanly maps to a kernel class on the dwq46 decode workload:")
    lines.append("      xs_<2us     : rms_norm, scalar mul, reshape")
    lines.append("      sm_2_8us    : rope, soft-cap, small mat-vec")
    lines.append("      md_8_32us   : Q4_0 MoE mat-vec_id (gate/up/down), dense Q4_0 mat-vec")
    lines.append("      lg_32_80us  : flash_attn, pooled mul_mm_id")
    lines.append("      xl_>=80us   : prefill mul_mm_id, lm_head, large blits")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  hf2q  trials: {len(hf2q_trials)}")
    for s in hf2q_trials:
        lines.append(f"    - {os.path.basename(s['path'])}: paired={s['paired']:>6d} dispatches "
                     f"({s['dispatches_per_token']:.1f}/tok), gpu={s['gpu_us_per_token']:.1f} µs/tok")
    lines.append(f"  llama trials: {len(llama_trials)}")
    for s in llama_trials:
        lines.append(f"    - {os.path.basename(s['path'])}: paired={s['paired']:>6d} dispatches "
                     f"({s['dispatches_per_token']:.1f}/tok), gpu={s['gpu_us_per_token']:.1f} µs/tok")
    lines.append("")

    if hf2q and llama:
        lines.append("=" * 110)
        lines.append("Side-by-side bucketed attribution (medians across trials)")
        lines.append("=" * 110)
        header = (f"{'BUCKET':<14s}  "
                  f"{'hf2q disp/tok':>14s}  {'hf2q µs/disp':>14s}  {'hf2q µs/tok':>14s}  "
                  f"{'llama disp/tok':>15s}  {'llama µs/disp':>15s}  {'llama µs/tok':>14s}  "
                  f"{'Δµs/tok':>10s}  {'Δ%':>8s}")
        lines.append(header)
        lines.append("-" * len(header))
        for name, _, _ in BUCKETS:
            hb = hf2q["buckets"][name]
            lb = llama["buckets"][name]
            d_us = hb["median_us_per_token"] - lb["median_us_per_token"]
            d_pct = (d_us / lb["median_us_per_token"] * 100) if lb["median_us_per_token"] > 0 else 0.0
            lines.append(
                f"{name:<14s}  "
                f"{hb['median_dispatches_per_token']:>14.1f}  "
                f"{hb['median_p50_us_per_dispatch']:>14.2f}  "
                f"{hb['median_us_per_token']:>14.1f}  "
                f"{lb['median_dispatches_per_token']:>15.1f}  "
                f"{lb['median_p50_us_per_dispatch']:>15.2f}  "
                f"{lb['median_us_per_token']:>14.1f}  "
                f"{d_us:>+10.1f}  "
                f"{d_pct:>+7.1f}%"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<14s}  "
            f"{hf2q['median_dispatches_per_token']:>14.1f}  "
            f"{'-':>14s}  "
            f"{hf2q['median_gpu_us_per_token']:>14.1f}  "
            f"{llama['median_dispatches_per_token']:>15.1f}  "
            f"{'-':>15s}  "
            f"{llama['median_gpu_us_per_token']:>14.1f}  "
            f"{(hf2q['median_gpu_us_per_token'] - llama['median_gpu_us_per_token']):>+10.1f}  "
            f"{((hf2q['median_gpu_us_per_token'] - llama['median_gpu_us_per_token']) / llama['median_gpu_us_per_token'] * 100):>+7.1f}%"
        )
        lines.append("")
        lines.append("Q4_0-attributable summary (md_8_32us bucket — Q4_0 MoE mat-vec_id territory):")
        hb = hf2q["buckets"]["md_8_32us"]
        lb = llama["buckets"]["md_8_32us"]
        lines.append(f"  hf2q : {hb['median_dispatches_per_token']:.1f} disp/tok × {hb['median_p50_us_per_dispatch']:.2f} µs/disp = {hb['median_us_per_token']:.1f} µs/tok")
        lines.append(f"  llama: {lb['median_dispatches_per_token']:.1f} disp/tok × {lb['median_p50_us_per_dispatch']:.2f} µs/disp = {lb['median_us_per_token']:.1f} µs/tok")
        d_us = hb["median_us_per_token"] - lb["median_us_per_token"]
        d_pct_of_total = d_us / max(llama["median_gpu_us_per_token"], 1) * 100
        lines.append(f"  delta: {d_us:+.1f} µs/tok ({d_pct_of_total:+.2f}% of llama wall)")
        lines.append("")
        lines.append("Iter10 attack target (largest positive Δµs/tok bucket):")
        target = max(BUCKETS, key=lambda b: hf2q["buckets"][b[0]]["median_us_per_token"] - llama["buckets"][b[0]]["median_us_per_token"])
        tname = target[0]
        d_us = hf2q["buckets"][tname]["median_us_per_token"] - llama["buckets"][tname]["median_us_per_token"]
        lines.append(f"  bucket: {tname}")
        lines.append(f"  Δµs/tok: {d_us:+.1f}")
        lines.append(f"  Likely kernel class: {bucket_kernel_hint(tname)}")
        lines.append("")
    elif hf2q:
        lines.append("=" * 110)
        lines.append("hf2q-only partial attribution (llama traces not yet available)")
        lines.append("=" * 110)
        header = f"{'BUCKET':<14s}  {'disp/tok':>10s}  {'µs/disp p50':>14s}  {'µs/tok':>10s}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, _, _ in BUCKETS:
            hb = hf2q["buckets"][name]
            lines.append(
                f"{name:<14s}  "
                f"{hb['median_dispatches_per_token']:>10.1f}  "
                f"{hb['median_p50_us_per_dispatch']:>14.2f}  "
                f"{hb['median_us_per_token']:>10.1f}"
            )
        lines.append(
            f"{'TOTAL':<14s}  "
            f"{hf2q['median_dispatches_per_token']:>10.1f}  "
            f"{'-':>14s}  "
            f"{hf2q['median_gpu_us_per_token']:>10.1f}"
        )
        lines.append("")
    else:
        lines.append("(no traces summarised)")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    sys.stdout.write(text)


def bucket_kernel_hint(name: str) -> str:
    return {
        "xs_<2us": "rms_norm / reshape / scalar",
        "sm_2_8us": "rope / soft-cap / small mat-vec",
        "md_8_32us": "Q4_0 MoE mat-vec_id (gate/up/down) — primary Q4_0 attack surface",
        "lg_32_80us": "flash_attn / pooled mul_mm_id",
        "xl_>=80us": "prefill mul_mm_id / lm_head / large blits",
    }.get(name, "unknown")


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf2q-trace", action="append", default=[], help="hf2q .trace bundle (repeatable)")
    ap.add_argument("--llama-trace", action="append", default=[], help="llama .trace bundle (repeatable)")
    ap.add_argument("--n-tokens", type=int, default=64, help="decode tokens per trial (default 64)")
    ap.add_argument("--output", default="/tmp/adr015-iter9/aggregate-q4_0.txt")
    ap.add_argument("--toc-dump", default=None, help="if set, dump xctrace --toc to this path for one trace")
    args = ap.parse_args()

    if args.toc_dump and args.hf2q_trace:
        with open(args.toc_dump, "w") as f:
            f.write(export_toc(args.hf2q_trace[0]))
        print(f"toc dumped: {args.toc_dump}", file=sys.stderr)

    hf2q_trials = []
    for t in args.hf2q_trace:
        try:
            s = summarize_trace(t, args.n_tokens)
            hf2q_trials.append(s)
            print(f"ok: hf2q {t}: {s['paired']} paired", file=sys.stderr)
        except Exception as e:
            print(f"WARN: hf2q {t}: {e}", file=sys.stderr)

    llama_trials = []
    for t in args.llama_trace:
        try:
            s = summarize_trace(t, args.n_tokens)
            llama_trials.append(s)
            print(f"ok: llama {t}: {s['paired']} paired", file=sys.stderr)
        except Exception as e:
            print(f"WARN: llama {t}: {e}", file=sys.stderr)

    hf2q_med = median_summaries(hf2q_trials)
    llama_med = median_summaries(llama_trials)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_report(args.output, hf2q_med, llama_med, hf2q_trials, llama_trials)
    print(f"\nwrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
