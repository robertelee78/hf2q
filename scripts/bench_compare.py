#!/usr/bin/env python3
"""
Compare hf2q and mlx-lm benchmark results.

Loads the JSON benchmark reports from both tools and generates a
side-by-side comparison table with pass/fail indicators.

Usage:
    python scripts/bench_compare.py --hf2q results/hf2q.json --mlx-lm results/mlx_lm.json
    python scripts/bench_compare.py --hf2q results/hf2q.json --mlx-lm results/mlx_lm.json --output comparison.md
"""

import argparse
import json
import sys


# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_report(path: str) -> dict:
    """Load a benchmark report from a JSON file."""
    with open(path) as f:
        return json.load(f)


def format_number(value: float, precision: int = 1) -> str:
    """Format a number with appropriate precision."""
    if value >= 1000:
        return f"{value:,.0f}"
    elif value >= 100:
        return f"{value:.0f}"
    elif value >= 10:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{precision + 1}f}"


def compare_metric(hf2q_val: float, mlx_val: float, higher_is_better: bool) -> tuple:
    """Compare two metric values.

    Returns (delta_pct, passed, symbol) where:
    - delta_pct: percentage difference (positive = hf2q is better)
    - passed: whether hf2q meets or exceeds mlx-lm
    - symbol: pass/fail indicator
    """
    if mlx_val == 0:
        return (0.0, True, "N/A")

    if higher_is_better:
        delta_pct = ((hf2q_val - mlx_val) / mlx_val) * 100
        passed = hf2q_val >= mlx_val
    else:
        delta_pct = ((mlx_val - hf2q_val) / mlx_val) * 100
        passed = hf2q_val <= mlx_val

    symbol = "PASS" if passed else "FAIL"
    return (delta_pct, passed, symbol)


def build_comparison_table(hf2q_report: dict, mlx_report: dict) -> list:
    """Build the comparison data structure."""
    rows = []

    # Match results by prompt length
    hf2q_by_len = {r["prompt_length"]: r for r in hf2q_report.get("results", [])}
    mlx_by_len = {r["prompt_length"]: r for r in mlx_report.get("results", [])}

    all_lengths = sorted(set(list(hf2q_by_len.keys()) + list(mlx_by_len.keys())))

    for prompt_len in all_lengths:
        hf2q_r = hf2q_by_len.get(prompt_len)
        mlx_r = mlx_by_len.get(prompt_len)

        if not hf2q_r or not mlx_r:
            continue

        # Decode tok/s (higher is better)
        hf2q_decode = hf2q_r["decode_tok_per_sec"]["median"]
        mlx_decode = mlx_r["decode_tok_per_sec"]["median"]
        delta_pct, passed, symbol = compare_metric(hf2q_decode, mlx_decode, True)
        rows.append({
            "prompt_len": prompt_len,
            "metric": "Decode tok/s",
            "hf2q": hf2q_decode,
            "mlx_lm": mlx_decode,
            "delta_pct": delta_pct,
            "passed": passed,
            "symbol": symbol,
        })

        # Prefill tok/s (higher is better)
        hf2q_prefill = hf2q_r["prefill_tok_per_sec"]["median"]
        mlx_prefill = mlx_r["prefill_tok_per_sec"]["median"]
        delta_pct, passed, symbol = compare_metric(hf2q_prefill, mlx_prefill, True)
        rows.append({
            "prompt_len": prompt_len,
            "metric": "Prefill tok/s",
            "hf2q": hf2q_prefill,
            "mlx_lm": mlx_prefill,
            "delta_pct": delta_pct,
            "passed": passed,
            "symbol": symbol,
        })

        # TTFT (lower is better)
        hf2q_ttft = hf2q_r["ttft_ms"]["median"]
        mlx_ttft = mlx_r["ttft_ms"]["median"]
        delta_pct, passed, symbol = compare_metric(hf2q_ttft, mlx_ttft, False)
        rows.append({
            "prompt_len": prompt_len,
            "metric": "TTFT (ms)",
            "hf2q": hf2q_ttft,
            "mlx_lm": mlx_ttft,
            "delta_pct": delta_pct,
            "passed": passed,
            "symbol": symbol,
        })

        # Peak memory (lower is better)
        hf2q_mem = hf2q_r.get("peak_memory_bytes")
        mlx_mem = mlx_r.get("peak_memory_bytes")
        if hf2q_mem is not None and mlx_mem is not None:
            delta_pct, passed, symbol = compare_metric(
                float(hf2q_mem), float(mlx_mem), False
            )
            rows.append({
                "prompt_len": prompt_len,
                "metric": "Peak memory (MB)",
                "hf2q": hf2q_mem / (1024 * 1024),
                "mlx_lm": mlx_mem / (1024 * 1024),
                "delta_pct": delta_pct,
                "passed": passed,
                "symbol": symbol,
            })

    # Model load time (lower is better)
    hf2q_load = hf2q_report.get("model_load_time_secs")
    mlx_load = mlx_report.get("model_load_time_secs")
    if hf2q_load is not None and mlx_load is not None:
        delta_pct, passed, symbol = compare_metric(hf2q_load, mlx_load, False)
        rows.append({
            "prompt_len": "all",
            "metric": "Model load (s)",
            "hf2q": hf2q_load,
            "mlx_lm": mlx_load,
            "delta_pct": delta_pct,
            "passed": passed,
            "symbol": symbol,
        })

    return rows


def print_terminal_table(rows: list, hf2q_report: dict, mlx_report: dict):
    """Print a colored comparison table to the terminal."""
    print(f"\n{BOLD}Benchmark Comparison: hf2q vs mlx-lm{RESET}")
    print(f"  hf2q model:  {hf2q_report.get('model', 'unknown')}")
    print(f"  mlx-lm model: {mlx_report.get('model', 'unknown')}")
    print()

    # Header
    header = f"{'Prompt':>6} | {'Metric':<18} | {'hf2q':>12} | {'mlx-lm':>12} | {'Delta':>8} | {'Result':>6}"
    print(header)
    print("-" * len(header))

    total_pass = 0
    total_fail = 0

    for row in rows:
        prompt_str = str(row["prompt_len"]).rjust(6)
        metric = row["metric"].ljust(18)
        hf2q_val = format_number(row["hf2q"])
        mlx_val = format_number(row["mlx_lm"])
        delta = f"{row['delta_pct']:+.1f}%"

        if row["passed"]:
            color = GREEN
            total_pass += 1
        else:
            color = RED
            total_fail += 1

        symbol = f"{color}{row['symbol']}{RESET}"
        print(f"{prompt_str} | {metric} | {hf2q_val:>12} | {mlx_val:>12} | {delta:>8} | {symbol}")

    print()
    pass_rate = total_pass / (total_pass + total_fail) * 100 if (total_pass + total_fail) > 0 else 0
    color = GREEN if total_fail == 0 else (YELLOW if pass_rate >= 70 else RED)
    print(f"{color}Results: {total_pass} passed, {total_fail} failed ({pass_rate:.0f}% pass rate){RESET}")
    print()


def generate_markdown(rows: list, hf2q_report: dict, mlx_report: dict) -> str:
    """Generate a Markdown comparison table."""
    lines = []
    lines.append("# Benchmark Comparison: hf2q vs mlx-lm\n")
    lines.append(f"- **hf2q model**: {hf2q_report.get('model', 'unknown')}")
    lines.append(f"- **mlx-lm model**: {mlx_report.get('model', 'unknown')}")
    lines.append(f"- **hf2q timestamp**: {hf2q_report.get('timestamp', 'unknown')}")
    lines.append(f"- **mlx-lm timestamp**: {mlx_report.get('timestamp', 'unknown')}")
    lines.append("")

    lines.append("| Prompt | Metric | hf2q | mlx-lm | Delta | Result |")
    lines.append("|--------|--------|------|--------|-------|--------|")

    for row in rows:
        hf2q_val = format_number(row["hf2q"])
        mlx_val = format_number(row["mlx_lm"])
        delta = f"{row['delta_pct']:+.1f}%"
        symbol = row["symbol"]
        emoji = "PASS" if row["passed"] else "**FAIL**"
        lines.append(
            f"| {row['prompt_len']} | {row['metric']} | {hf2q_val} | {mlx_val} | {delta} | {emoji} |"
        )

    lines.append("")

    total_pass = sum(1 for r in rows if r["passed"])
    total_fail = sum(1 for r in rows if not r["passed"])
    lines.append(f"**Results**: {total_pass} passed, {total_fail} failed")
    lines.append("")

    # Methodology section
    methodology = hf2q_report.get("methodology", {})
    lines.append("## Methodology\n")
    lines.append(f"- Warm-up runs: {methodology.get('warm_up_runs', 'N/A')}")
    lines.append(f"- Measurement runs: {methodology.get('measurement_runs', 'N/A')}")
    lines.append(f"- Statistic: {methodology.get('statistic', 'N/A')}")
    lines.append(f"- Max tokens per run: {methodology.get('max_tokens_per_run', 'N/A')}")
    lines.append(f"- Prompt generation: {methodology.get('prompt_generation', 'N/A')}")
    lines.append("")

    for note in methodology.get("notes", []):
        lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare hf2q and mlx-lm benchmark results"
    )
    parser.add_argument(
        "--hf2q", required=True,
        help="Path to hf2q benchmark JSON report"
    )
    parser.add_argument(
        "--mlx-lm", required=True,
        help="Path to mlx-lm benchmark JSON report"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output Markdown file (default: terminal output only)"
    )

    args = parser.parse_args()

    hf2q_report = load_report(args.hf2q)
    mlx_report = load_report(getattr(args, "mlx_lm"))

    rows = build_comparison_table(hf2q_report, mlx_report)

    if not rows:
        print("Error: No comparable results found between the two reports.",
              file=sys.stderr)
        sys.exit(1)

    # Terminal output
    print_terminal_table(rows, hf2q_report, mlx_report)

    # Markdown output
    if args.output:
        md = generate_markdown(rows, hf2q_report, mlx_report)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Markdown report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
