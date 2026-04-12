#!/usr/bin/env bash
# phase0_candle_bench.sh — ADR-006 Phase 0 per-kernel GPU instrumentation bench.
#
# Runs hf2q with the MTLCounterSampleBuffer instrumentation ON for 5
# consecutive invocations (each generating 128 tokens from the canonical
# prompt at T=0 greedy), then runs 5 uninstrumented control invocations
# to validate the observer-effect gate (median tok/s within +/-2% of
# the 84.9 tok/s post-1bNEW.22 baseline).
#
# The instrumented dump file is cumulatively written to
# docs/phase0-candle-perkernel-raw.json. The final structured output
# conforming to the Phase 0 schema is written to
# docs/phase0-candle-perkernel.json by a Python post-processing step.
#
# Usage:
#   scripts/phase0_candle_bench.sh <gguf_path>
#
# Prerequisites:
#   - target/release/hf2q built with --features metal
#   - Python 3 (for JSON post-processing and median computation)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GGUF_PATH="${1:-models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf}"
HF2Q_BIN="target/release/hf2q"
PROMPT_FILE="tests/bench_prompt_128.txt"
MAX_TOKENS=128
NUM_RUNS=5
BASELINE_TOKS=84.9    # post-1bNEW.22 reference
GATE_PCT=2.0           # +/-2% observer-effect gate
RAW_DUMP="docs/phase0-candle-perkernel-raw.json"
OUTPUT_JSON="docs/phase0-candle-perkernel.json"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

err() { echo "error: $*" >&2; exit 1; }

[[ -f "$GGUF_PATH" ]]   || err "GGUF not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN" ]]    || err "hf2q binary not found: $HF2Q_BIN (run: cargo build --release --features metal)"
[[ -f "$PROMPT_FILE" ]]  || err "Prompt file not found: $PROMPT_FILE"
command -v python3 >/dev/null 2>&1 || err "python3 is required"

echo "=== ADR-006 Phase 0 — per-kernel GPU instrumentation bench ==="
echo "GGUF:        $GGUF_PATH"
echo "hf2q:        $HF2Q_BIN"
echo "git HEAD:    $GIT_HEAD"
echo "prompt:      $PROMPT_FILE"
echo "max_tokens:  $MAX_TOKENS"
echo "runs:        $NUM_RUNS instrumented + $NUM_RUNS uninstrumented"
echo

# Check thermal state
THERMAL=$(pmset -g therm 2>/dev/null | grep -oE 'CPU_Scheduler_Limit = [0-9]+' | grep -oE '[0-9]+' || echo "unknown")
if [[ "$THERMAL" == "100" ]]; then
    THERMAL_NOTE="idle (100%)"
elif [[ "$THERMAL" != "unknown" ]]; then
    THERMAL_NOTE="throttled ($THERMAL%)"
else
    THERMAL_NOTE="unknown"
fi
echo "Thermal state: $THERMAL_NOTE"
echo

# === Phase A: Instrumented runs ===
echo "--- Phase A: $NUM_RUNS instrumented runs ---"
rm -f "$RAW_DUMP"
INST_TOKS=()

for run in $(seq 1 "$NUM_RUNS"); do
    echo -n "  Run $run/$NUM_RUNS: "
    # Each invocation runs the full prefill+decode and the instrument
    # module's atexit hook dumps the cumulative aggregator to RAW_DUMP.
    OUTPUT=$(HF2Q_PHASE0_INSTRUMENT=1 \
             HF2Q_PHASE0_INSTRUMENT_DUMP="$RAW_DUMP" \
             "$HF2Q_BIN" generate \
               --model "$GGUF_PATH" \
               --prompt-file "$PROMPT_FILE" \
               --max-tokens "$MAX_TOKENS" \
               --temperature 0 \
               2>&1 || true)

    # Parse tok/s from the "--- N tokens in X.XXs (Y.Y tok/s) ---" line
    TPS=$(echo "$OUTPUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | awk '{print $1}')
    if [[ -z "$TPS" ]]; then
        echo "FAILED (no tok/s in output)"
        continue
    fi
    INST_TOKS+=("$TPS")
    echo "$TPS tok/s"
done

echo

# === Phase B: Uninstrumented control runs ===
echo "--- Phase B: $NUM_RUNS uninstrumented control runs ---"
CTRL_TOKS=()

for run in $(seq 1 "$NUM_RUNS"); do
    echo -n "  Run $run/$NUM_RUNS: "
    OUTPUT=$("$HF2Q_BIN" generate \
               --model "$GGUF_PATH" \
               --prompt-file "$PROMPT_FILE" \
               --max-tokens "$MAX_TOKENS" \
               --temperature 0 \
               2>&1 || true)

    TPS=$(echo "$OUTPUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | awk '{print $1}')
    if [[ -z "$TPS" ]]; then
        echo "FAILED (no tok/s in output)"
        continue
    fi
    CTRL_TOKS+=("$TPS")
    echo "$TPS tok/s"
done

echo

# === Phase C: Post-processing ===
echo "--- Phase C: Post-processing ---"

# Build the final JSON via Python.
INST_LIST=$(printf '%s,' "${INST_TOKS[@]}" | sed 's/,$//')
CTRL_LIST=$(printf '%s,' "${CTRL_TOKS[@]}" | sed 's/,$//')

python3 - "$RAW_DUMP" "$OUTPUT_JSON" "$GIT_HEAD" "$THERMAL_NOTE" "$BASELINE_TOKS" "$GATE_PCT" "$MAX_TOKENS" "$NUM_RUNS" "$INST_LIST" "$CTRL_LIST" <<'PYEOF'
import json, sys, statistics, os

raw_path, output_path = sys.argv[1], sys.argv[2]
git_head = sys.argv[3]
thermal_note = sys.argv[4]
baseline_toks = float(sys.argv[5])
gate_pct = float(sys.argv[6])
max_tokens = int(sys.argv[7])
n_runs = int(sys.argv[8])
inst_toks = [float(x) for x in sys.argv[9].split(',') if x]
ctrl_toks = [float(x) for x in sys.argv[10].split(',') if x]

inst_median = statistics.median(inst_toks) if inst_toks else 0.0
ctrl_median = statistics.median(ctrl_toks) if ctrl_toks else 0.0
delta_pct = abs(ctrl_median - baseline_toks) / baseline_toks * 100 if baseline_toks > 0 else 999

with open(raw_path) as f:
    raw = json.load(f)

kernels = raw.get("kernels", [])
overflow = raw.get("overflow_count", 0)
error_values = raw.get("error_value_count", 0)

# Per-kernel: compute us/call median, p95, calls_per_token
kernel_rows = []
for k in kernels:
    calls = k["calls"]
    median_ns = k["median_ns"]
    p95_ns = k["p95_ns"]
    # calls_per_token: approximate by dividing by (n_runs * max_tokens)
    cpt = calls / (n_runs * max_tokens) if (n_runs * max_tokens) > 0 else 0.0
    us_per_call_median = median_ns / 1000.0
    us_per_call_p95 = p95_ns / 1000.0
    us_per_token = us_per_call_median * cpt
    kernel_rows.append({
        "name": k["name"],
        "calls_per_token_median": round(cpt, 2),
        "us_per_call_median": round(us_per_call_median, 2),
        "us_per_call_p95": round(us_per_call_p95, 2),
        "us_per_token_median": round(us_per_token, 2),
        "dispatch_shapes_seen": [],
        "total_calls": calls,
        "total_ns": k["total_ns"],
        "mean_ns": k["mean_ns"],
        "samples_recorded": k.get("samples_recorded", calls)
    })

# Sort by us_per_token descending.
kernel_rows.sort(key=lambda r: r["us_per_token_median"], reverse=True)
top5 = [r["name"] for r in kernel_rows[:5]]

result = {
    "stack": "hf2q+candle",
    "commit": git_head,
    "device": "M5 Max",
    "model": "Gemma 4 26B MoE Q4_K_M",
    "prompt": "tests/bench_prompt_128.txt",
    "n_tokens": max_tokens,
    "temperature": 0,
    "n_runs": n_runs,
    "wall_clock_tok_per_sec_median": round(inst_median, 1),
    "wall_clock_ms_per_token_median": round(1000.0 / inst_median, 2) if inst_median > 0 else 0,
    "thermal_state_notes": thermal_note,
    "kernels": kernel_rows,
    "top5_by_us_per_token": top5,
    "observer_effect_check": {
        "uninstrumented_tok_per_sec_median": round(ctrl_median, 1),
        "delta_pct": round(delta_pct, 2),
        "within_2pct_of_baseline": delta_pct <= gate_pct
    },
    "instrumentation_notes": {
        "overflow_count": overflow,
        "error_value_count": error_values,
        "sample_buffer_slots": 4096,
        "note": "overflow_count > 0 means some dispatches were uninstrumented due to sample buffer exhaustion between flush_and_wait calls. Per-kernel medians remain statistically valid; absolute totals should be scaled by (total_dispatches / instrumented_dispatches)."
    }
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
    f.write('\n')

print(f"Instrumented median: {inst_median:.1f} tok/s ({n_runs} runs)")
print(f"Uninstrumented median: {ctrl_median:.1f} tok/s ({n_runs} runs)")
print(f"Observer-effect delta: {delta_pct:.2f}% vs {baseline_toks:.1f} tok/s baseline")
print(f"Observer-effect gate: {'PASS' if delta_pct <= gate_pct else 'FAIL'} (threshold: +/-{gate_pct}%)")
print(f"Overflow count: {overflow}")
print(f"Top 5 kernels by us/token:")
for r in kernel_rows[:5]:
    print(f"  {r['name']}: {r['us_per_token_median']:.1f} us/token ({r['calls_per_token_median']:.1f} calls/token, {r['us_per_call_median']:.1f} us/call)")
print(f"Output: {output_path}")
PYEOF

echo
echo "--- Results ---"
echo "Instrumented runs:   ${INST_TOKS[*]}"
echo "Uninstrumented runs: ${CTRL_TOKS[*]}"
echo "Output: $OUTPUT_JSON"
echo "Raw dump: $RAW_DUMP"
echo
echo "Done."
