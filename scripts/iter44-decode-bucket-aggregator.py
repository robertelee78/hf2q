#!/usr/bin/env python3
"""
ADR-015 iter44 decode-bucket aggregator.

Parses HF2Q_DECODE_PROFILE=1 stderr from `hf2q generate --benchmark` and
emits per-token µs/tok averages for each profile bucket:

  - linear_attn  (DeltaNet layers' attention)
  - full_attn    (FullAttention layers' attention)
  - ffn          (MoE/Dense FFN)
  - output_head  (norm + lm_head + argmax + commit_and_wait sync time)
  - cmd_bufs     (count per token)
  - dispatches   (count per token)
  - total_layers (sum of attn+ffn+norm+residual; CPU-encoding wall)

Skips the first GREEDY_PROFILE line (warmup / first-decode-token outliers
include init / cache state), and all DECODE_PROFILE lines (those are
prefill, not decode).

Usage:
  python3 scripts/iter44-decode-bucket-aggregator.py <stderr_file> [<stderr_file2> ...]

If multiple stderr files are passed, prints per-trial summary then a
multi-trial median/mean table.
"""

import re
import sys
from statistics import mean, median, stdev


GREEDY_LINE = re.compile(
    r"\[GREEDY_PROFILE\] linear_attn=(?P<la>[\d.]+)ms "
    r"full_attn=(?P<fa>[\d.]+)ms "
    r"ffn=(?P<ffn>[\d.]+)ms "
    r"total_layers=(?P<tl>[\d.]+)ms "
    r"cmd_bufs=(?P<cb>\d+) "
    r"dispatches=(?P<disp>\d+)"
)
OUTPUT_HEAD_LINE = re.compile(
    r"\[GREEDY_PROFILE\] output_head=(?P<oh>[\d.]+)ms"
)
DECODE_TPS = re.compile(r"Decode tok/s: ([\d.]+)")
DECODE_WALL = re.compile(r"--- mlx-native[^:]*: (\d+) tokens in ([\d.]+)s")


def parse_one(path: str, *, skip_first: int = 1) -> dict:
    """Aggregate one stderr (+stdout if available)."""
    la, fa, ffn, tl, cb, disp, oh = ([] for _ in range(7))
    decode_tps = None
    decode_wall_s = None
    n_tokens = None
    # We zip GREEDY layer lines with following output_head lines (1:1).
    # Track them in two lists, drop indices < skip_first.
    layer_lines = []
    head_lines = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m1 = GREEDY_LINE.search(line)
            m2 = OUTPUT_HEAD_LINE.search(line)
            if m1:
                layer_lines.append({
                    "la": float(m1.group("la")),
                    "fa": float(m1.group("fa")),
                    "ffn": float(m1.group("ffn")),
                    "tl": float(m1.group("tl")),
                    "cb": int(m1.group("cb")),
                    "disp": int(m1.group("disp")),
                })
            elif m2:
                head_lines.append(float(m2.group("oh")))
            md = DECODE_TPS.search(line)
            if md:
                decode_tps = float(md.group(1))
            mw = DECODE_WALL.search(line)
            if mw:
                n_tokens = int(mw.group(1))
                decode_wall_s = float(mw.group(2))

    # Stdout file may have decode metadata when stderr doesn't.
    stdout_path = path.replace(".stderr", ".stdout")
    if decode_tps is None:
        try:
            with open(stdout_path, "r", errors="replace") as f:
                for line in f:
                    md = DECODE_TPS.search(line)
                    if md:
                        decode_tps = float(md.group(1))
                    mw = DECODE_WALL.search(line)
                    if mw:
                        n_tokens = int(mw.group(1))
                        decode_wall_s = float(mw.group(2))
        except FileNotFoundError:
            pass

    layer_lines = layer_lines[skip_first:]
    head_lines = head_lines[skip_first:]
    n = min(len(layer_lines), len(head_lines))
    layer_lines = layer_lines[:n]
    head_lines = head_lines[:n]

    if n == 0:
        return {
            "path": path, "n_decode_tokens": 0, "decode_tps": decode_tps,
            "decode_wall_s": decode_wall_s, "raw_n_tokens": n_tokens,
        }

    sum_la = sum(x["la"] for x in layer_lines)
    sum_fa = sum(x["fa"] for x in layer_lines)
    sum_ffn = sum(x["ffn"] for x in layer_lines)
    sum_tl = sum(x["tl"] for x in layer_lines)
    sum_oh = sum(head_lines)
    sum_us = sum_la + sum_fa + sum_ffn + sum_oh
    cb_typical = layer_lines[0]["cb"]
    disp_typical = layer_lines[0]["disp"]

    return {
        "path": path,
        "n_decode_tokens": n,
        "skip_first": skip_first,
        "decode_tps": decode_tps,
        "decode_wall_s": decode_wall_s,
        "raw_n_tokens": n_tokens,
        # per-token µs/tok (= ms/tok × 1000)
        "us_la_per_tok": (sum_la / n) * 1000,
        "us_fa_per_tok": (sum_fa / n) * 1000,
        "us_ffn_per_tok": (sum_ffn / n) * 1000,
        "us_tl_per_tok": (sum_tl / n) * 1000,
        "us_oh_per_tok": (sum_oh / n) * 1000,
        "us_layers_plus_head_per_tok": ((sum_tl + sum_oh) / n) * 1000,
        "cmd_bufs_per_tok": cb_typical,
        "dispatches_per_tok": disp_typical,
        # implied per-token wall (1/tps)
        "us_wall_per_tok_implied": (1e6 / decode_tps) if decode_tps else None,
    }


def fmt_us(v):
    if v is None:
        return "-"
    return f"{v:8.1f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = sys.argv[1:]
    results = [parse_one(p) for p in paths]

    print(f"{'trial':<24} {'n':>4} {'tps':>7} {'wall_us':>9} "
          f"{'la_us':>9} {'fa_us':>9} {'ffn_us':>9} "
          f"{'tl_us':>9} {'oh_us':>9} {'tl+oh_us':>10} "
          f"{'cb':>4} {'disp':>5}")
    for r in results:
        if r["n_decode_tokens"] == 0:
            print(f"{r['path']:<24} EMPTY")
            continue
        print(
            f"{r['path'].split('/')[-1][:24]:<24} "
            f"{r['n_decode_tokens']:>4} "
            f"{r['decode_tps']:>7.2f} "
            f"{fmt_us(r['us_wall_per_tok_implied'])} "
            f"{fmt_us(r['us_la_per_tok'])} "
            f"{fmt_us(r['us_fa_per_tok'])} "
            f"{fmt_us(r['us_ffn_per_tok'])} "
            f"{fmt_us(r['us_tl_per_tok'])} "
            f"{fmt_us(r['us_oh_per_tok'])} "
            f"{fmt_us(r['us_layers_plus_head_per_tok'])} "
            f"{r['cmd_bufs_per_tok']:>4} "
            f"{r['dispatches_per_tok']:>5}"
        )

    if len(results) > 1:
        valid = [r for r in results if r["n_decode_tokens"] > 0]
        if valid:
            print()
            print("=== multi-trial summary (median across trials) ===")
            keys = [
                ("decode_tps", "tps"),
                ("us_wall_per_tok_implied", "wall_us"),
                ("us_la_per_tok", "la_us"),
                ("us_fa_per_tok", "fa_us"),
                ("us_ffn_per_tok", "ffn_us"),
                ("us_tl_per_tok", "tl_us"),
                ("us_oh_per_tok", "oh_us"),
                ("us_layers_plus_head_per_tok", "tl+oh_us"),
            ]
            for key, lbl in keys:
                vals = [r[key] for r in valid if r[key] is not None]
                if not vals:
                    continue
                m = median(vals)
                avg = mean(vals)
                sd = stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {lbl:<14}  median={m:9.2f}  mean={avg:9.2f}  stdev={sd:7.2f}  n={len(vals)}")


if __name__ == "__main__":
    main()
