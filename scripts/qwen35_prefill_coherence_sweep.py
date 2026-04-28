#!/usr/bin/env python3
"""Qwen3.6 prefill speed + coherence sweep.

This is a measure-first harness for prefill work.  It calibrates prompts by the
actual tokenizer, then for each prompt length runs:

1. hf2q baseline timing.
2. hf2q candidate timing.
3. hf2q baseline/candidate last-prefill-logit dumps and numeric comparison.
4. Optional llama-completion prompt-eval timing as a peer reference.

The key rule: a candidate is not a speed win unless its logits/top tokens remain
inside the configured coherence thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer


DEFAULT_MODEL = (
    "/opt/hf2q/models/qwen3.6-27b-dwq46/"
    "qwen3.6-27b-dwq46.gguf"
)
DEFAULT_TOKENIZER = "/opt/hf2q/models/qwen3.6-27b-dwq46/tokenizer.json"
DEFAULT_HF2Q = "/opt/hf2q/target/release/hf2q"
DEFAULT_LLAMA = "/opt/homebrew/bin/llama-completion"
CHAT_TEMPLATE_RAW = '{{ messages[0]["content"] }}'

BASE_TEXT = (
    "The benchmark prompt describes a careful engineering investigation. "
    "It repeats neutral facts about measuring model speed, preserving token "
    "probabilities, checking coherence, and comparing current code against "
    "peer implementations. "
)


def parse_env(items: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"bad env item {item!r}; expected NAME=VALUE")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def token_count(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text).ids)


def make_prompt(tok: Tokenizer, target_tokens: int) -> tuple[str, int]:
    words = (BASE_TEXT * ((target_tokens // 20) + 200)).split()
    lo, hi = 1, len(words)
    best = " ".join(words[:lo])
    best_n = token_count(tok, best)
    while lo <= hi:
        mid = (lo + hi) // 2
        text = " ".join(words[:mid])
        n = token_count(tok, text)
        if n <= target_tokens:
            best, best_n = text, n
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_n


def run(cmd: list[str], env_extra: dict[str, str], stdout: Path, stderr: Path) -> int:
    env = os.environ.copy()
    env.update(env_extra)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)
    return proc.returncode


def parse_hf2q_timing(stdout: Path, stderr: Path) -> dict[str, float | int | None]:
    text = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
    prefill = re.search(r"prefill:\s+(\d+) tok in ([0-9.]+)ms \(([0-9.]+) tok/s\)", text)
    first = re.search(r"first decoded token:\s+(\d+)", text)
    return {
        "prefill_tokens": int(prefill.group(1)) if prefill else None,
        "prefill_ms": float(prefill.group(2)) if prefill else None,
        "prefill_tps": float(prefill.group(3)) if prefill else None,
        "first_token": int(first.group(1)) if first else None,
    }


def parse_llama_timing(stdout: Path, stderr: Path) -> dict[str, float | int | None]:
    text = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
    m = re.search(
        r"prompt eval time =\s+([0-9.]+) ms /\s+(\d+) tokens.*?([0-9.]+) tokens per second",
        text,
    )
    return {
        "llama_prefill_ms": float(m.group(1)) if m else None,
        "llama_prefill_tokens": int(m.group(2)) if m else None,
        "llama_prefill_tps": float(m.group(3)) if m else None,
    }


def read_f32_file(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise RuntimeError(f"{path} length {len(data)} is not f32-aligned")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def topk(vals: list[float], k: int) -> list[tuple[int, float]]:
    return sorted(enumerate(vals), key=lambda x: x[1], reverse=True)[:k]


def compare_logits(base_path: Path, cand_path: Path) -> dict[str, object]:
    base = read_f32_file(base_path)
    cand = read_f32_file(cand_path)
    if len(base) != len(cand):
        raise RuntimeError(f"logit length mismatch: {len(base)} vs {len(cand)}")
    max_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    b2 = 0.0
    c2 = 0.0
    for b, c in zip(base, cand):
        d = abs(b - c)
        if d > max_abs:
            max_abs = d
        sum_sq += d * d
        dot += b * c
        b2 += b * b
        c2 += c * c
    rms = math.sqrt(sum_sq / len(base))
    cosine = dot / math.sqrt(b2 * c2) if b2 > 0 and c2 > 0 else float("nan")
    bt = topk(base, 10)
    ct = topk(cand, 10)
    return {
        "max_abs_logit_delta": max_abs,
        "rms_logit_delta": rms,
        "cosine": cosine,
        "base_top1": bt[0][0],
        "candidate_top1": ct[0][0],
        "top10_overlap": len({i for i, _ in bt} & {i for i, _ in ct}),
        "base_top10": bt,
        "candidate_top10": ct,
    }


def hf2q_cmd(args: argparse.Namespace, prompt: Path, max_tokens: int) -> list[str]:
    return [
        args.hf2q_bin,
        "-v",
        "generate",
        "--model",
        args.model,
        "--prompt-file",
        str(prompt),
        "--chat-template",
        CHAT_TEMPLATE_RAW,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--hf2q-bin", default=DEFAULT_HF2Q)
    ap.add_argument("--llama-bin", default=DEFAULT_LLAMA)
    ap.add_argument("--out-dir", default="/tmp/qwen35-prefill-coherence-sweep")
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--base-env", action="append", default=[])
    ap.add_argument("--candidate-env", action="append", default=[])
    ap.add_argument("--run-llama", action="store_true")
    ap.add_argument(
        "--max-abs-threshold",
        type=float,
        default=1.25,
        help=(
            "Allowed max absolute last-prefill-logit drift. Default is the "
            "2026-04-28 repeatability envelope observed on identical qwen35 "
            "runs at 512-4096 tokens; tighten after prefill nondeterminism is fixed."
        ),
    )
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--min-top10-overlap", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    base_env = parse_env(args.base_env)
    cand_env = parse_env(args.candidate_env)
    common_env = {
        "HF2Q_QWEN36_AUTOREG": "1",
        "HF2Q_UNSAFE_EXPERIMENTS": "1",
        "HF2Q_CHUNK_SCAN_PREFILL": "1",
    }
    base_env = {**common_env, **base_env}
    cand_env = {**common_env, **cand_env}

    rows = []
    failed = False

    for target in lengths:
        prompt_text, actual = make_prompt(tok, target)
        prompt_path = out_dir / f"prompt-{target}.txt"
        prompt_path.write_text(prompt_text)

        row: dict[str, object] = {"target_tokens": target, "tokenizer_tokens": actual}

        for label, env in [("base", base_env), ("candidate", cand_env)]:
            stdout = out_dir / f"{label}-{target}.stdout"
            stderr = out_dir / f"{label}-{target}.stderr"
            rc = run(hf2q_cmd(args, prompt_path, 2), env, stdout, stderr)
            row[f"{label}_rc"] = rc
            row.update({f"{label}_{k}": v for k, v in parse_hf2q_timing(stdout, stderr).items()})

            dump_stdout = out_dir / f"{label}-{target}-dump.stdout"
            dump_stderr = out_dir / f"{label}-{target}-dump.stderr"
            tmp_logits = Path("/tmp/hf2q_logits_t0.bin")
            if tmp_logits.exists():
                tmp_logits.unlink()
            dump_env = {**env, "HF2Q_DUMP_LOGITS": "1"}
            rc = run(hf2q_cmd(args, prompt_path, 1), dump_env, dump_stdout, dump_stderr)
            row[f"{label}_dump_rc"] = rc
            dest = out_dir / f"{label}-{target}-logits.bin"
            if tmp_logits.exists():
                shutil.move(str(tmp_logits), dest)
            else:
                row[f"{label}_dump_missing"] = True

        base_logits = out_dir / f"base-{target}-logits.bin"
        cand_logits = out_dir / f"candidate-{target}-logits.bin"
        if base_logits.exists() and cand_logits.exists():
            cmp = compare_logits(base_logits, cand_logits)
            row.update(cmp)
            if cmp["base_top1"] != cmp["candidate_top1"]:
                failed = True
                row["gate"] = "FAIL_TOP1"
            elif cmp["top10_overlap"] < args.min_top10_overlap:
                failed = True
                row["gate"] = "FAIL_TOP10"
            elif cmp["max_abs_logit_delta"] > args.max_abs_threshold:
                failed = True
                row["gate"] = "FAIL_MAX_ABS"
            elif cmp["cosine"] < args.min_cosine:
                failed = True
                row["gate"] = "FAIL_COSINE"
            else:
                row["gate"] = "PASS"
        else:
            failed = True
            row["gate"] = "FAIL_MISSING_LOGITS"

        if args.run_llama:
            lout = out_dir / f"llama-{target}.stdout"
            lerr = out_dir / f"llama-{target}.stderr"
            llama_cmd = [
                args.llama_bin,
                "--model",
                args.model,
                "--batch-size",
                "4096",
                "--ubatch-size",
                "512",
                "--n-predict",
                "1",
                "--no-warmup",
                "-no-cnv",
                "--perf",
                "-f",
                str(prompt_path),
            ]
            row["llama_rc"] = run(llama_cmd, {}, lout, lerr)
            row.update(parse_llama_timing(lout, lerr))

        base_ms = row.get("base_prefill_ms")
        cand_ms = row.get("candidate_prefill_ms")
        llama_ms = row.get("llama_prefill_ms")
        if isinstance(base_ms, float) and isinstance(cand_ms, float) and cand_ms > 0:
            row["candidate_speedup_vs_base"] = base_ms / cand_ms
        if isinstance(base_ms, float) and isinstance(llama_ms, float) and llama_ms > 0:
            row["base_speed_vs_llama"] = llama_ms / base_ms
        if isinstance(cand_ms, float) and isinstance(llama_ms, float) and llama_ms > 0:
            row["candidate_speed_vs_llama"] = llama_ms / cand_ms

        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = out_dir / "summary.jsonl"
    with summary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"summary: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
