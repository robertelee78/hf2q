#!/usr/bin/env python3
"""Warm-cache Qwen3.6 prefill sweep.

This wrapper drives the production `hf2q generate` qwen35 path once with
`HF2Q_QWEN35_PREFILL_SWEEP` set.  The Rust process loads the model once, runs
explicit shape warmups, then runs measured prefill trials in-process.  With
`--profile`, live W5B profiler buckets are parsed from stderr and attached to
each JSONL row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


DEFAULT_MODEL = (
    "/opt/hf2q/models/qwen3.6-27b-dwq46/"
    "qwen3.6-27b-dwq46.gguf"
)
DEFAULT_HF2Q = "/opt/hf2q/target/release/hf2q"
PROFILE_ROW = re.compile(
    r"^\[W5B8_PROFILE\]\s+([A-Za-z0-9_.]+)\s+(\d+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)


def parse_profiles(stderr_path: Path) -> list[dict[str, dict[str, float | int]]]:
    profiles: list[dict[str, dict[str, float | int]]] = []
    current: dict[str, dict[str, float | int]] | None = None
    for line in stderr_path.read_text(errors="replace").splitlines():
        if line.startswith("[W5B8_PROFILE] === section summary:"):
            current = {}
            continue
        if line.startswith("[W5B8_PROFILE] === end summary ==="):
            if current is not None:
                profiles.append(current)
            current = None
            continue
        if current is None:
            continue
        m = PROFILE_ROW.match(line)
        if not m:
            continue
        current[m.group(1)] = {
            "n": int(m.group(2)),
            "sum_ms": float(m.group(3)),
            "mean_ms": float(m.group(4)),
            "min_ms": float(m.group(5)),
            "max_ms": float(m.group(6)),
            "p50_ms": float(m.group(7)),
            "p95_ms": float(m.group(8)),
        }
    return profiles


def parse_env_items(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --env {item!r}; expected NAME=VALUE")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf2q-bin", default=DEFAULT_HF2Q)
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--full-logits",
        action="store_true",
        help="Use the legacy full [seq_len,vocab] output head instead of last-token logits.",
    )
    ap.add_argument(
        "--compare-full-last",
        action="store_true",
        help="Run full and last output heads in-process and compare the final logit row.",
    )
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out-dir", default="/tmp/qwen35-prefill-warm-sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "hf2q-warm-sweep.stdout"
    stderr_path = out_dir / "hf2q-warm-sweep.stderr"
    summary_path = out_dir / "summary.jsonl"

    env = os.environ.copy()
    env.update(
        {
            "HF2Q_QWEN36_AUTOREG": "1",
            "HF2Q_UNSAFE_EXPERIMENTS": "1",
            "HF2Q_CHUNK_SCAN_PREFILL": "1",
            "HF2Q_QWEN35_PREFILL_SWEEP": args.lengths,
            "HF2Q_QWEN35_PREFILL_SWEEP_WARMUPS": str(args.warmups),
            "HF2Q_QWEN35_PREFILL_SWEEP_TRIALS": str(args.trials),
        }
    )
    if args.profile:
        env.update(
            {
                "HF2Q_PROFILE_W5B8": "1",
                "HF2Q_PROFILE_W5B17": "1",
                "HF2Q_PROFILE_W5B22": "1",
            }
        )
    if args.full_logits:
        env["HF2Q_QWEN35_PREFILL_SWEEP_FULL_LOGITS"] = "1"
    if args.compare_full_last:
        env["HF2Q_QWEN35_PREFILL_SWEEP_COMPARE_FULL_LAST"] = "1"
    env.update(parse_env_items(args.env))

    cmd = [
        args.hf2q_bin,
        "-v",
        "generate",
        "--model",
        args.model,
        "--prompt",
        "unused by HF2Q_QWEN35_PREFILL_SWEEP",
        "--max-tokens",
        "1",
        "--temperature",
        "0",
    ]
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)

    rows = []
    compare_rows = []
    for line in stdout_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "qwen35_prefill_sweep":
            rows.append(obj)
        elif obj.get("event") == "qwen35_prefill_sweep_compare":
            compare_rows.append(obj)

    profiles = parse_profiles(stderr_path) if args.profile else []
    if profiles:
        if len(profiles) != len(rows):
            print(
                f"warning: parsed {len(profiles)} W5B profiles for {len(rows)} sweep rows",
                flush=True,
            )
        for row, profile in zip(rows, profiles):
            row["sections"] = profile

    with summary_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print("target actual phase trial head prefill_ms prefill_tps first_token ffn_ms chunk_call_ms full_ms")
    for row in rows:
        sections = row.get("sections", {})
        ffn = sections.get("layer.ffn_dispatch", {}).get("sum_ms") if isinstance(sections, dict) else None
        chunk = sections.get("layer.chunk_call", {}).get("sum_ms") if isinstance(sections, dict) else None
        full = sections.get("layer.full_total", {}).get("sum_ms") if isinstance(sections, dict) else None
        print(
            row["target_tokens"],
            row["actual_tokens"],
            row.get("phase", "measure"),
            row["trial"],
            row.get("output_head"),
            row["prefill_ms"],
            row["prefill_tps"],
            row["first_token"],
            ffn,
            chunk,
            full,
        )
    if compare_rows:
        print("compare target actual phase trial full_token last_token max_abs cosine top10_overlap")
        for row in compare_rows:
            print(
                row["target_tokens"],
                row["actual_tokens"],
                row.get("phase", "measure"),
                row["trial"],
                row["full_token"],
                row["last_token"],
                row["max_abs"],
                row["cosine"],
                row["top10_overlap"],
            )
    print(f"summary: {summary_path}")
    print(f"stdout:  {stdout_path}")
    print(f"stderr:  {stderr_path}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
