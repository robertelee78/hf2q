#!/usr/bin/env python3
"""
ADR-015 iter45-RESUMED — N-curve results aggregator.

Reads /tmp/adr015-iter45/bench/<fixture>/{llama-base,cn-N}/<DATE>.* and
emits the final per-fixture × per-N table for the ADR §iter45-RESUMED
PHASE 4 deliverable.

For each fixture × cn cell:
  - hf2q tps median, p10, p90 across N trials (parsed via iter44 aggregator)
  - llama tps median (5 trials at llama-base, independent of cn)
  - ratio = hf2q_med / llama_med
  - n_cb measured (from iter44 aggregator's `cb` line per-trial)

Usage:
  python3 scripts/iter45-aggregate-results.py [--out_dir DIR] [--date_tag TAG]

Default OUT_DIR: /tmp/adr015-iter45/bench
Default DATE_TAG: latest by glob mtime.
"""

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys


GREEDY_LINE = re.compile(
    r"\[GREEDY_PROFILE\] linear_attn=([\d.]+)ms "
    r"full_attn=([\d.]+)ms "
    r"ffn=([\d.]+)ms "
    r"total_layers=([\d.]+)ms "
    r"cmd_bufs=(\d+) "
    r"dispatches=(\d+)"
)
DECODE_TPS = re.compile(r"Decode tok/s: ([\d.]+)")
LLAMA_TG = re.compile(r"\|\s*tg(\d+)\s*\|\s*([\d.]+)\s*±")


def parse_hf2q_trial(stderr_path: str):
    """Return dict with tps, cb_typical from one stderr file (skipping first decode token)."""
    cb_vals = []
    tps = None
    layer_n = 0
    stdout_path = stderr_path.replace(".stderr", ".stdout")
    for path in (stderr_path, stdout_path):
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    m = GREEDY_LINE.search(line)
                    if m:
                        layer_n += 1
                        if layer_n > 1:  # skip first decode token
                            cb_vals.append(int(m.group(5)))
                    md = DECODE_TPS.search(line)
                    if md:
                        tps = float(md.group(1))
        except FileNotFoundError:
            pass
    cb = cb_vals[0] if cb_vals else None  # cb is constant per model/cn; first is fine
    return {"tps": tps, "cb": cb, "n_decode": layer_n}


def parse_llama_trial(stdout_path: str, ngen: int):
    """Return llama tg<ngen> tps from llama-bench stdout."""
    try:
        with open(stdout_path, "r", errors="replace") as f:
            for line in f:
                m = LLAMA_TG.search(line)
                if m and int(m.group(1)) == ngen:
                    return float(m.group(2))
    except FileNotFoundError:
        pass
    return None


def percentiles(vals, p):
    """Robust percentile (linear interpolation; handles small N)."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/adr015-iter45/bench")
    ap.add_argument("--date_tag", default=None,
                    help="defaults to latest DATE_TAG found by glob")
    ap.add_argument("--ngen", type=int, default=256)
    ap.add_argument("--fixtures", default="dwq46,apex,27b-dwq46,gemma-26B")
    ap.add_argument("--ns", default="1,2,4,8,20")
    args = ap.parse_args()

    fixtures = args.fixtures.split(",")
    ns = [int(n) for n in args.ns.split(",")]

    # Auto-detect DATE_TAG if not given
    date_tag = args.date_tag
    if date_tag is None:
        candidates = []
        for f in fixtures:
            for stub in glob.glob(os.path.join(args.out_dir, f, "llama-base", "*.llama.trial-1.stdout")):
                bn = os.path.basename(stub)
                # bn like "20260429T190141Z.llama.trial-1.stdout"
                tag = bn.split(".")[0]
                candidates.append((os.path.getmtime(stub), tag))
        if not candidates:
            print(f"ERROR: no llama-base trial-1.stdout found under {args.out_dir}",
                  file=sys.stderr)
            sys.exit(1)
        candidates.sort()
        date_tag = candidates[-1][1]
    print(f"DATE_TAG = {date_tag}")
    print(f"NGEN = {args.ngen}")
    print()

    print(f"{'fixture':<14} {'cn':>3} {'hf2q_med':>10} {'hf2q_p10':>10} "
          f"{'hf2q_p90':>10} {'llama_med':>10} {'ratio':>8} {'n_cb':>5} {'n_trials':>9}")
    print("-" * 90)

    summary_rows = []
    for fixture in fixtures:
        # Llama base (one set of trials per fixture)
        llama_dir = os.path.join(args.out_dir, fixture, "llama-base")
        llama_tps_vals = []
        for f in sorted(glob.glob(os.path.join(llama_dir, f"{date_tag}.llama.trial-*.stdout"))):
            v = parse_llama_trial(f, args.ngen)
            if v is not None:
                llama_tps_vals.append(v)
        llama_med = statistics.median(llama_tps_vals) if llama_tps_vals else None

        for cn in ns:
            cell_dir = os.path.join(args.out_dir, fixture, f"cn-{cn}")
            stderr_files = sorted(glob.glob(os.path.join(cell_dir, f"{date_tag}.hf2q.trial-*.stderr")))
            if not stderr_files:
                summary_rows.append({
                    "fixture": fixture, "cn": cn,
                    "hf2q_med": None, "hf2q_p10": None, "hf2q_p90": None,
                    "llama_med": llama_med, "ratio": None, "cb": None, "n": 0,
                })
                continue
            tps_vals, cb_vals = [], []
            for f in stderr_files:
                t = parse_hf2q_trial(f)
                if t["tps"] is not None:
                    tps_vals.append(t["tps"])
                if t["cb"] is not None:
                    cb_vals.append(t["cb"])
            hf2q_med = statistics.median(tps_vals) if tps_vals else None
            hf2q_p10 = percentiles(tps_vals, 10) if tps_vals else None
            hf2q_p90 = percentiles(tps_vals, 90) if tps_vals else None
            ratio = (hf2q_med / llama_med) if hf2q_med and llama_med else None
            cb = cb_vals[0] if cb_vals else None  # cb is const per cn
            summary_rows.append({
                "fixture": fixture, "cn": cn,
                "hf2q_med": hf2q_med, "hf2q_p10": hf2q_p10, "hf2q_p90": hf2q_p90,
                "llama_med": llama_med, "ratio": ratio, "cb": cb, "n": len(tps_vals),
            })

    for r in summary_rows:
        def fmt(v, w=10, p=2):
            return f"{v:{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}"
        ratio_str = f"{r['ratio']:.4f}" if r['ratio'] is not None else "-"
        cb_str = str(r['cb']) if r['cb'] is not None else "-"
        print(
            f"{r['fixture']:<14} {r['cn']:>3} "
            f"{fmt(r['hf2q_med'])} {fmt(r['hf2q_p10'])} {fmt(r['hf2q_p90'])} "
            f"{fmt(r['llama_med'])} {ratio_str:>8} {cb_str:>5} {r['n']:>9}"
        )

    # Markdown table for ADR § (one block per fixture)
    print()
    print("=" * 92)
    print("MARKDOWN TABLES (one per fixture) — paste into ADR-015 §iter45-RESUMED:")
    print("=" * 92)
    for fixture in fixtures:
        rows = [r for r in summary_rows if r["fixture"] == fixture]
        if not rows:
            continue
        print(f"\n### Fixture: {fixture}")
        print()
        print("| chain_n N | hf2q t/s med | hf2q p10/p90 | llama t/s med | ratio  | n_cb measured |")
        print("|-----------|-------------:|-------------:|--------------:|-------:|--------------:|")
        for r in rows:
            def f(v, p=2):
                return f"{v:.{p}f}" if isinstance(v, float) else "-"
            p10p90 = (
                f"{r['hf2q_p10']:.2f} / {r['hf2q_p90']:.2f}"
                if r['hf2q_p10'] is not None and r['hf2q_p90'] is not None
                else "-"
            )
            ratio = f"{r['ratio']:.4f}" if r['ratio'] is not None else "-"
            cb = str(r['cb']) if r['cb'] is not None else "-"
            print(f"| {r['cn']:>9} | {f(r['hf2q_med']):>12} | {p10p90:>12} | "
                  f"{f(r['llama_med']):>13} | {ratio:>6} | {cb:>13} |")


if __name__ == "__main__":
    main()
