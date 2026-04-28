#!/usr/bin/env python3
"""Qwen3.6 prefill bucket profiler.

Runs hf2q with the live W5B profile gates and parses the actual
`[W5B8_PROFILE]` summary emitted by the Rust code.  The output is JSONL so the
results can be compared across prompt lengths without trusting comments or ADRs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

PROFILE_ROW = re.compile(
    r"^\[W5B8_PROFILE\]\s+([A-Za-z0-9_.]+)\s+(\d+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
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
    """Create a stable prompt near target_tokens using tokenizer round trips."""
    repeated = BASE_TEXT * ((target_tokens // 20) + 500)
    ids = tok.encode(repeated).ids
    if len(ids) < target_tokens:
        raise RuntimeError(f"base prompt only produced {len(ids)} tokens")

    best_text = tok.decode(ids[:target_tokens])
    best_n = token_count(tok, best_text)
    if best_n == target_tokens:
        return best_text, best_n

    lo = max(1, target_tokens - 256)
    hi = min(len(ids), target_tokens + 256)
    for n_ids in range(lo, hi + 1):
        text = tok.decode(ids[:n_ids])
        n = token_count(tok, text)
        if n == target_tokens:
            return text, n
        if abs(n - target_tokens) < abs(best_n - target_tokens):
            best_text, best_n = text, n

    return best_text, best_n


def run(cmd: list[str], env_extra: dict[str, str], stdout: Path, stderr: Path) -> int:
    env = os.environ.copy()
    env.update(env_extra)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)
    return proc.returncode


def parse_hf2q(stdout: Path, stderr: Path) -> dict[str, object]:
    text = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
    prefill = re.search(r"prefill:\s+(\d+) tok in ([0-9.]+)ms \(([0-9.]+) tok/s\)", text)
    first = re.search(r"first decoded token:\s+(\d+)", text)
    chunk_engaged = "chunk-pipeline ENGAGED" in text
    chunk_failed = "chunk-pipeline gate set but predicate FAILED" in text
    sections: dict[str, dict[str, float | int]] = {}
    for line in text.splitlines():
        m = PROFILE_ROW.match(line)
        if not m:
            continue
        sections[m.group(1)] = {
            "n": int(m.group(2)),
            "sum_ms": float(m.group(3)),
            "mean_ms": float(m.group(4)),
            "min_ms": float(m.group(5)),
            "max_ms": float(m.group(6)),
            "p50_ms": float(m.group(7)),
            "p95_ms": float(m.group(8)),
        }
    return {
        "prefill_tokens": int(prefill.group(1)) if prefill else None,
        "prefill_ms": float(prefill.group(2)) if prefill else None,
        "prefill_tps": float(prefill.group(3)) if prefill else None,
        "first_token": int(first.group(1)) if first else None,
        "chunk_engaged": chunk_engaged,
        "chunk_failed": chunk_failed,
        "sections": sections,
    }


def parse_llama(stdout: Path, stderr: Path) -> dict[str, object]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--hf2q-bin", default=DEFAULT_HF2Q)
    ap.add_argument("--llama-bin", default=DEFAULT_LLAMA)
    ap.add_argument("--out-dir", default="/tmp/qwen35-prefill-bucket-profile")
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--run-llama", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    env = {
        "HF2Q_QWEN36_AUTOREG": "1",
        "HF2Q_UNSAFE_EXPERIMENTS": "1",
        "HF2Q_CHUNK_SCAN_PREFILL": "1",
        "HF2Q_PROFILE_W5B8": "1",
        "HF2Q_PROFILE_W5B17": "1",
        "HF2Q_PROFILE_W5B22": "1",
    }
    env.update(parse_env(args.env))

    rows: list[dict[str, object]] = []
    failed = False
    for target in lengths:
        prompt_text, tokenizer_tokens = make_prompt(tok, target)
        prompt_path = out_dir / f"prompt-{target}.txt"
        prompt_path.write_text(prompt_text)

        stdout = out_dir / f"hf2q-{target}.stdout"
        stderr = out_dir / f"hf2q-{target}.stderr"
        cmd = [
            args.hf2q_bin,
            "-v",
            "generate",
            "--model",
            args.model,
            "--prompt-file",
            str(prompt_path),
            "--chat-template",
            CHAT_TEMPLATE_RAW,
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ]
        rc = run(cmd, env, stdout, stderr)
        parsed = parse_hf2q(stdout, stderr)
        row: dict[str, object] = {
            "target_tokens": target,
            "tokenizer_tokens": tokenizer_tokens,
            "hf2q_rc": rc,
            **parsed,
        }
        if rc != 0:
            failed = True

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
            row.update(parse_llama(lout, lerr))

        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = out_dir / "summary.jsonl"
    with summary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print("\ncompact bucket table:")
    print(
        "target actual hf2q_ms llama_ms chunk linear_ms full_ms ffn_ms "
        "chunk_call_ms fa_sdpa_ms qkv_split_ms"
    )
    for row in rows:
        sections = row.get("sections", {})
        assert isinstance(sections, dict)

        def sec_ms(name: str) -> object:
            sec = sections.get(name)
            return sec.get("sum_ms") if isinstance(sec, dict) else None

        print(
            row.get("target_tokens"),
            row.get("prefill_tokens"),
            row.get("prefill_ms"),
            row.get("llama_prefill_ms"),
            "engaged" if row.get("chunk_engaged") else ("failed" if row.get("chunk_failed") else "unknown"),
            sec_ms("layer.linear_total"),
            sec_ms("layer.full_total"),
            sec_ms("layer.ffn_dispatch"),
            sec_ms("layer.chunk_call"),
            sec_ms("fa.sdpa_total"),
            sec_ms("dn.qkv_gpu_split"),
        )
    print(f"summary: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
