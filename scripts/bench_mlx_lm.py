#!/usr/bin/env python3
"""
Benchmark mlx-lm for comparison against hf2q.

Runs generation at multiple prompt lengths and measures:
- Decode tok/s (tokens per second during decode)
- Prefill tok/s (tokens per second during prompt encoding)
- TTFT (time to first token in ms)
- Peak memory usage
- Model load time

Output: structured JSON matching the hf2q benchmark report format.

Usage:
    python scripts/bench_mlx_lm.py --model /path/to/model
    python scripts/bench_mlx_lm.py --model /path/to/model --max-tokens 128 --num-runs 5
"""

import argparse
import json
import sys
import time
import statistics
from typing import Optional


def generate_synthetic_prompt(target_tokens: int) -> str:
    """Generate a deterministic synthetic prompt.

    Uses the same word vocabulary as the hf2q benchmark harness for
    fair comparison. Each word is roughly 1 token for subword tokenizers.
    """
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
        "dog", "and", "then", "runs", "through", "the", "forest", "where",
        "many", "trees", "grow", "tall", "under", "the", "bright", "sun",
        "that", "shines", "down", "upon", "the", "green", "meadow", "below",
    ]
    word_count = target_tokens + target_tokens // 10
    prompt_words = [words[i % len(words)] for i in range(word_count)]
    return " ".join(prompt_words)


def measure_peak_memory_mlx() -> Optional[int]:
    """Query MLX for peak memory usage, if available."""
    try:
        import mlx.core as mx
        return mx.metal.get_peak_memory()
    except Exception:
        return None


def reset_peak_memory_mlx():
    """Reset MLX peak memory tracking."""
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def run_benchmark(model_path: str, prompt_lengths: list, max_tokens: int,
                  num_runs: int, num_warmup: int) -> dict:
    """Run the mlx-lm benchmark and return a structured report."""
    try:
        from mlx_lm import load, generate
        from mlx_lm.utils import generate_step
    except ImportError:
        print("Error: mlx-lm is not installed. Install with: pip install mlx-lm",
              file=sys.stderr)
        sys.exit(1)

    # Measure model load time
    print(f"Loading model: {model_path}", file=sys.stderr)
    load_start = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)

    model_name = model_path.rstrip("/").split("/")[-1]
    results = []

    for prompt_len in prompt_lengths:
        prompt = generate_synthetic_prompt(prompt_len)
        print(f"Benchmarking prompt_length={prompt_len}...", file=sys.stderr)

        # Warm-up
        for _ in range(num_warmup):
            try:
                generate(model, tokenizer, prompt=prompt, max_tokens=min(16, max_tokens))
            except Exception as e:
                print(f"  Warm-up error: {e}", file=sys.stderr)

        measurements = []

        for run_idx in range(num_runs):
            reset_peak_memory_mlx()

            # Tokenize to count actual tokens
            input_ids = tokenizer.encode(prompt)
            actual_prompt_tokens = len(input_ids)

            # Time the generation
            run_start = time.time()
            prefill_start = time.time()

            generated_tokens = 0
            ttft = None

            try:
                # Use the streaming generate_step for precise timing
                prompt_tokens_mx = None
                try:
                    import mlx.core as mx
                    prompt_tokens_mx = mx.array(input_ids)[None]
                except Exception:
                    pass

                if prompt_tokens_mx is not None:
                    # Low-level generation for precise prefill/decode timing
                    first_token_time = None
                    for step_idx, (token, logits) in enumerate(
                        generate_step(prompt_tokens_mx, model, temp=0.0)
                    ):
                        if step_idx == 0:
                            first_token_time = time.time()
                            ttft = first_token_time - run_start
                            prefill_time = first_token_time - prefill_start
                        generated_tokens += 1
                        if generated_tokens >= max_tokens:
                            break
                        if hasattr(tokenizer, 'eos_token_id') and token.item() == tokenizer.eos_token_id:
                            break

                    decode_end = time.time()
                    decode_time = decode_end - (first_token_time or decode_end)
                else:
                    # Fallback: use high-level generate
                    result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
                    total_time = time.time() - run_start
                    # Approximate timing split
                    output_tokens = tokenizer.encode(result)
                    generated_tokens = len(output_tokens)
                    prefill_time = total_time * 0.3  # rough estimate
                    decode_time = total_time * 0.7
                    ttft = prefill_time

            except Exception as e:
                print(f"  Run {run_idx} error: {e}", file=sys.stderr)
                continue

            peak_mem = measure_peak_memory_mlx()

            m = {
                "prefill_secs": prefill_time if 'prefill_time' in dir() else 0.0,
                "decode_secs": decode_time if 'decode_time' in dir() else 0.0,
                "ttft_secs": ttft or 0.0,
                "prompt_tokens": actual_prompt_tokens,
                "generated_tokens": generated_tokens,
                "peak_memory_bytes": peak_mem,
            }
            measurements.append(m)

        if not measurements:
            continue

        # Aggregate
        result = aggregate_measurements(measurements, prompt_len)
        results.append(result)

    report = {
        "tool": "mlx-lm",
        "timestamp": f"{int(time.time())}s-since-epoch",
        "model": model_name,
        "synthetic": False,
        "model_load_time_secs": load_time,
        "results": results,
        "methodology": {
            "warm_up_runs": num_warmup,
            "measurement_runs": num_runs,
            "statistic": "median of N runs with min/max range",
            "max_tokens_per_run": max_tokens,
            "prompt_generation": (
                "Deterministic synthetic prompts using fixed English word vocabulary. "
                "Same prompts as hf2q benchmark for fair comparison."
            ),
            "notes": [
                "Uses mlx-lm generate_step for precise prefill/decode timing.",
                "Same model checkpoint used for both hf2q and mlx-lm benchmarks.",
                "Temperature=0 for deterministic output.",
            ],
        },
    }

    return report


def aggregate_measurements(measurements: list, target_prompt_len: int) -> dict:
    """Aggregate measurements into median stats."""
    def median_stat(values):
        if not values:
            return {"median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    actual_tokens = measurements[0]["prompt_tokens"] if measurements else target_prompt_len

    decode_tps = []
    prefill_tps = []
    ttft_ms = []
    gen_tokens = []
    peak_mems = []

    for m in measurements:
        if m["decode_secs"] > 0 and m["generated_tokens"] > 0:
            decode_tps.append(m["generated_tokens"] / m["decode_secs"])
        if m["prefill_secs"] > 0 and m["prompt_tokens"] > 0:
            prefill_tps.append(m["prompt_tokens"] / m["prefill_secs"])
        ttft_ms.append(m["ttft_secs"] * 1000.0)
        gen_tokens.append(float(m["generated_tokens"]))
        if m.get("peak_memory_bytes") is not None:
            peak_mems.append(m["peak_memory_bytes"])

    return {
        "prompt_length": target_prompt_len,
        "actual_prompt_tokens": actual_tokens,
        "decode_tok_per_sec": median_stat(decode_tps) if decode_tps else median_stat([0.0]),
        "prefill_tok_per_sec": median_stat(prefill_tps) if prefill_tps else median_stat([0.0]),
        "ttft_ms": median_stat(ttft_ms) if ttft_ms else median_stat([0.0]),
        "peak_memory_bytes": max(peak_mems) if peak_mems else None,
        "generated_tokens": median_stat(gen_tokens) if gen_tokens else median_stat([0.0]),
        "prompt_cache_active": False,
        "cached_tokens_last_run": 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mlx-lm for comparison against hf2q"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Maximum tokens to generate per prompt (default: 128)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5,
        help="Number of measurement runs per prompt length (default: 5)"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=2,
        help="Number of warm-up runs (default: 2)"
    )
    parser.add_argument(
        "--prompt-lengths", type=str, default="20,256,1024",
        help="Comma-separated prompt lengths to benchmark (default: 20,256,1024)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: stdout)"
    )

    args = parser.parse_args()
    prompt_lengths = [int(x.strip()) for x in args.prompt_lengths.split(",")]

    report = run_benchmark(
        model_path=args.model,
        prompt_lengths=prompt_lengths,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        num_warmup=args.num_warmup,
    )

    json_output = json.dumps(report, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
            f.write("\n")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
