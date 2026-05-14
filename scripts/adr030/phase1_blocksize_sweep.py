#!/usr/bin/env python3
"""ADR-030 Phase 1.5 — block_size sweep on the two most-distinct prompts.

H_BLOCK_SIZE hypothesis: smaller K reduces reject-waste per step, so
the explainer prompt (15.2% accept) should recover more than the math
prompt (39.9% accept) when K shrinks.

Smaller-budget bench: 2 K values × 2 prompts × 3 cycles × 2 arms = 24
measurements ≈ 50 min. Math + explainer chosen as extremes from the
Phase 1 full-bench (1.187× best, 0.461× worst); code's 1.014× sat
between them, so adding it doesn't change which side wins/loses.

Output: docs/research/ADR-030-phase1-blocksize-sweep.json
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
    ("math", "How many positive whole-number divisors does 196 have?"),
    ("explainer", "Explain in 3 short paragraphs how Flash Attention reduces memory in attention computation."),
]
MAX_TOKENS = 256
BLOCK_SIZES = [8, 12]  # K=7 / K=11; baseline K=15 already measured in phase1_run1
CYCLES = 3
COOL_DOWN_SEC = 60


def _cool(sec):
    print(f"[cool {sec}s]", flush=True)
    time.sleep(sec)


def bench_baseline(model, tok, prompt_str, max_tokens, temp=0.0):
    sampler = make_sampler(temp=temp)
    n = 0
    last_r = None
    tic = time.perf_counter()
    for r in mlxlm_stream(model, tok, prompt_str, max_tokens=max_tokens, sampler=sampler):
        last_r = r
        n += 1
    elapsed = time.perf_counter() - tic
    tps = getattr(last_r, "generation_tps", n / elapsed) if last_r else (n / elapsed)
    return {"arm": "baseline", "tps": float(tps), "tokens": int(n), "elapsed_sec": float(elapsed)}


def bench_dflash(model, draft, tok, prompt_str, max_tokens, block_size, temp=0.0):
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
    tps = last_r.generation_tps if last_r else 0.0
    mean_accepted = statistics.mean(accepted_steps) if accepted_steps else 0.0
    return {
        "arm": "dflash",
        "tps": float(tps),
        "tokens": int(total_tokens),
        "elapsed_sec": float(elapsed),
        "mean_accepted_per_step": float(mean_accepted),
        "n_steps": int(len(accepted_steps)),
        "block_size": int(block_size),
    }


def build_prompt_str(tok, user_prompt):
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def warmup(model, draft, tok, block_size):
    p = build_prompt_str(tok, "Say hello.")
    print(f"[warmup K={block_size}]", flush=True)
    bench_dflash(model, draft, tok, p, max_tokens=32, block_size=block_size)
    _cool(15)


def run(args):
    print(f"[device] {mx.device_info()}", flush=True)
    print(f"[load] target={args.target}", flush=True)
    model, tok = load(args.target)
    print(f"[load] draft={args.draft}", flush=True)
    draft = load_draft(args.draft)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "spec": "ADR-030 Phase 1.5 block-size sweep",
        "target": args.target,
        "draft": args.draft,
        "block_sizes": BLOCK_SIZES,
        "cycles": CYCLES,
        "cool_down_sec": COOL_DOWN_SEC,
        "max_tokens": MAX_TOKENS,
        "samples": [],
    }

    for block_size in BLOCK_SIZES:
        warmup(model, draft, tok, block_size)
        for prompt_name, prompt in PROMPTS:
            prompt_str = build_prompt_str(tok, prompt)
            for cycle in range(CYCLES):
                _cool(COOL_DOWN_SEC)
                a = bench_baseline(model, tok, prompt_str, max_tokens=MAX_TOKENS)
                a.update({"prompt": prompt_name, "cycle": cycle, "block_size": block_size})
                _cool(COOL_DOWN_SEC)
                b = bench_dflash(model, draft, tok, prompt_str, max_tokens=MAX_TOKENS, block_size=block_size)
                b.update({"prompt": prompt_name, "cycle": cycle})
                ratio = b["tps"] / a["tps"] if a["tps"] > 0 else 0.0
                print(f"K={block_size} prompt={prompt_name} cycle={cycle} "
                      f"baseline={a['tps']:.2f} dflash={b['tps']:.2f} "
                      f"accept/step={b['mean_accepted_per_step']:.2f} speedup={ratio:.2f}x", flush=True)
                results["samples"].extend([a, b])
                out_path.write_text(json.dumps(results, indent=2))

    # Summary
    summary = []
    for block_size in BLOCK_SIZES:
        for prompt_name, _ in PROMPTS:
            base = [s["tps"] for s in results["samples"]
                    if s["arm"] == "baseline" and s["prompt"] == prompt_name and s["block_size"] == block_size]
            dfl = [s["tps"] for s in results["samples"]
                   if s["arm"] == "dflash" and s["prompt"] == prompt_name and s["block_size"] == block_size]
            accs = [s["mean_accepted_per_step"] for s in results["samples"]
                    if s["arm"] == "dflash" and s["prompt"] == prompt_name and s["block_size"] == block_size]
            if base and dfl:
                summary.append({
                    "block_size": block_size,
                    "prompt": prompt_name,
                    "baseline_mean": statistics.mean(base),
                    "dflash_mean": statistics.mean(dfl),
                    "speedup": statistics.mean(dfl) / statistics.mean(base),
                    "mean_accepted_per_step": statistics.mean(accs) if accs else 0.0,
                })
    results["summary"] = summary
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=TARGET_ID)
    p.add_argument("--draft", default=DRAFT_ID)
    p.add_argument("--out", default="/opt/hf2q/docs/research/ADR-030-phase1-blocksize-sweep.json")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
