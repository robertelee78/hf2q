#!/usr/bin/env python3
"""ADR-030 Phase 1 — Standalone DFlash validation on M5 Max.

NOT part of hf2q runtime. Python sidecar that consumes the z-lab
DFlash reference implementation at /opt/dflash to MEASURE whether the
expected 1.6-3x speedup over single-token decode materializes on our
target hardware (M5 Max + gemma-4-26B-A4B-it) BEFORE we commit to the
~1,810 LOC Rust port in Phase 2+.

Bench protocol (per feedback_metal_bench_protocol_2026_05_12):
  - Alt-pair thermal-fair: baseline + dflash interleaved per cycle
  - 60s cool-downs between every measurement
  - 5 cycles minimum at 3 distinct prompts
  - Single hf2q/dflash instance at a time (feedback_one_instance_at_a_time)
  - max_tokens fixed (no early-EOS) to mirror --ignore-eos discipline

Output: docs/research/ADR-030-phase1-m5max-results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from dflash.model_mlx import load, load_draft, stream_generate as dflash_stream
from mlx_lm.generate import stream_generate as mlxlm_stream
from mlx_lm.sample_utils import make_sampler

TARGET_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
DRAFT_ID = "z-lab/gemma-4-26B-A4B-it-DFlash"
PROMPTS = [
    "How many positive whole-number divisors does 196 have?",
    "Explain in 3 short paragraphs how Flash Attention reduces memory in attention computation.",
    "Write a Rust function that computes the nth Fibonacci number iteratively.",
]
MAX_TOKENS = 256
DEFAULT_CYCLES = 5
DEFAULT_COOL_DOWN_SEC = 60
DEFAULT_BLOCK_SIZE = 16


def _cool(sec: int) -> None:
    print(f"[cool {sec}s]", flush=True)
    time.sleep(sec)


def bench_baseline(model, tok, prompt_str: str, max_tokens: int, temp: float = 0.0) -> dict:
    sampler = make_sampler(temp=temp)
    n = 0
    last_r = None
    tic = time.perf_counter()
    for r in mlxlm_stream(model, tok, prompt_str, max_tokens=max_tokens, sampler=sampler):
        last_r = r
        n += 1
    elapsed = time.perf_counter() - tic
    # mlx_lm stream_generate's GenerationResponse has generation_tps directly
    tps = getattr(last_r, "generation_tps", n / elapsed) if last_r is not None else (n / elapsed)
    return {
        "arm": "baseline",
        "tps": float(tps),
        "tokens": int(n),
        "elapsed_sec": float(elapsed),
        "peak_mem_gb": float(mx.get_peak_memory() / 1e9),
    }


def bench_dflash(model, draft, tok, prompt_str: str, max_tokens: int, block_size: int, temp: float = 0.0) -> dict:
    last_r = None
    accepted_steps = []
    total_tokens = 0
    tic = time.perf_counter()
    for r in dflash_stream(model, draft, tok, prompt_str,
                            block_size=block_size, max_tokens=max_tokens, temperature=temp):
        last_r = r
        if r.accepted > 0:
            accepted_steps.append(r.accepted)
        total_tokens = r.generation_tokens
    elapsed = time.perf_counter() - tic
    tps = last_r.generation_tps if last_r is not None else 0.0
    mean_accepted = statistics.mean(accepted_steps) if accepted_steps else 0.0
    return {
        "arm": "dflash",
        "tps": float(tps),
        "tokens": int(total_tokens),
        "elapsed_sec": float(elapsed),
        "peak_mem_gb": float(last_r.peak_memory if last_r else 0.0),
        "mean_accepted_per_step": float(mean_accepted),
        "n_steps": int(len(accepted_steps)),
        "block_size": int(block_size),
    }


def build_prompt_str(tok, user_prompt: str) -> str:
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def warmup(model, draft, tok) -> None:
    """One short warmup of each arm to seed the PSO cache + thermal floor.
    Per feedback_metal_bench_protocol_2026_05_12: σ < 1% precondition.
    """
    p = build_prompt_str(tok, "Say hello.")
    print("[warmup baseline]", flush=True)
    bench_baseline(model, tok, p, max_tokens=32)
    _cool(15)
    print("[warmup dflash]", flush=True)
    bench_dflash(model, draft, tok, p, max_tokens=32, block_size=DEFAULT_BLOCK_SIZE)
    _cool(15)


def run(args) -> None:
    target_id = args.target
    draft_id = args.draft
    cycles = args.cycles
    cool_sec = args.cool
    max_tokens = args.max_tokens
    block_size = args.block_size

    print(f"[device] {mx.device_info()}", flush=True)
    print(f"[load] target={target_id}", flush=True)
    tic = time.perf_counter()
    model, tok = load(target_id)
    print(f"[load] target loaded in {time.perf_counter()-tic:.1f}s", flush=True)
    tic = time.perf_counter()
    print(f"[load] draft={draft_id}", flush=True)
    draft = load_draft(draft_id)
    print(f"[load] draft loaded in {time.perf_counter()-tic:.1f}s", flush=True)

    warmup(model, draft, tok)

    results = {
        "spec": "ADR-030 Phase 1 M5 Max baseline",
        "target": target_id,
        "draft": draft_id,
        "cycles": cycles,
        "cool_down_sec": cool_sec,
        "max_tokens": max_tokens,
        "block_size": block_size,
        "device": dict(mx.device_info()),
        "prompts": PROMPTS,
        "samples": [],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for prompt_idx, prompt in enumerate(PROMPTS):
        prompt_str = build_prompt_str(tok, prompt)
        for cycle in range(cycles):
            _cool(cool_sec)
            a = bench_baseline(model, tok, prompt_str, max_tokens=max_tokens)
            a["prompt_idx"] = prompt_idx
            a["cycle"] = cycle

            _cool(cool_sec)
            b = bench_dflash(model, draft, tok, prompt_str, max_tokens=max_tokens, block_size=block_size)
            b["prompt_idx"] = prompt_idx
            b["cycle"] = cycle

            ratio = b["tps"] / a["tps"] if a["tps"] > 0 else 0.0
            print(
                f"prompt={prompt_idx} cycle={cycle} "
                f"baseline={a['tps']:.2f} t/s  dflash={b['tps']:.2f} t/s  "
                f"accept/step={b['mean_accepted_per_step']:.2f}  "
                f"speedup={ratio:.2f}x",
                flush=True,
            )
            results["samples"].extend([a, b])
            # Atomic-ish write: rewrite the full results file each cycle so
            # interrupted runs leave usable data.
            out_path.write_text(json.dumps(results, indent=2))

    # Summary stats per prompt + overall
    summary = {"per_prompt": [], "overall": {}}
    for prompt_idx in range(len(PROMPTS)):
        base_tps = [s["tps"] for s in results["samples"]
                    if s["arm"] == "baseline" and s["prompt_idx"] == prompt_idx]
        dfl_tps = [s["tps"] for s in results["samples"]
                   if s["arm"] == "dflash" and s["prompt_idx"] == prompt_idx]
        accs = [s["mean_accepted_per_step"] for s in results["samples"]
                if s["arm"] == "dflash" and s["prompt_idx"] == prompt_idx]
        if base_tps and dfl_tps:
            summary["per_prompt"].append({
                "prompt_idx": prompt_idx,
                "baseline_mean": statistics.mean(base_tps),
                "baseline_stdev": statistics.stdev(base_tps) if len(base_tps) > 1 else 0.0,
                "baseline_sigma_pct": (statistics.stdev(base_tps) / statistics.mean(base_tps) * 100) if (len(base_tps) > 1 and statistics.mean(base_tps) > 0) else 0.0,
                "dflash_mean": statistics.mean(dfl_tps),
                "dflash_stdev": statistics.stdev(dfl_tps) if len(dfl_tps) > 1 else 0.0,
                "dflash_sigma_pct": (statistics.stdev(dfl_tps) / statistics.mean(dfl_tps) * 100) if (len(dfl_tps) > 1 and statistics.mean(dfl_tps) > 0) else 0.0,
                "speedup_mean": statistics.mean(dfl_tps) / statistics.mean(base_tps),
                "mean_accepted_per_step": statistics.mean(accs) if accs else 0.0,
            })
    all_base = [s["tps"] for s in results["samples"] if s["arm"] == "baseline"]
    all_dfl = [s["tps"] for s in results["samples"] if s["arm"] == "dflash"]
    all_acc = [s["mean_accepted_per_step"] for s in results["samples"] if s["arm"] == "dflash"]
    if all_base and all_dfl:
        summary["overall"] = {
            "baseline_mean": statistics.mean(all_base),
            "dflash_mean": statistics.mean(all_dfl),
            "speedup_mean": statistics.mean(all_dfl) / statistics.mean(all_base),
            "mean_accepted_per_step": statistics.mean(all_acc) if all_acc else 0.0,
            "go_no_go_gate_1_6x": (statistics.mean(all_dfl) / statistics.mean(all_base)) >= 1.6,
        }
    results["summary"] = summary
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=TARGET_ID)
    p.add_argument("--draft", default=DRAFT_ID)
    p.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    p.add_argument("--cool", type=int, default=DEFAULT_COOL_DOWN_SEC)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--out", default="/opt/hf2q/docs/research/ADR-030-phase1-m5max-results.json")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
