#!/usr/bin/env bash
# ADR-034 P6 — Multi-Token-Prediction (MTP) paired tok/s bench.
#
# Measures MTP-enabled (HF2Q_SPEC_DECODE=1) vs baseline (HF2Q_SPEC_DECODE=0)
# decode throughput on a target GGUF. Alt-pair scheduling: spec, base, spec,
# base, ... — controls for thermal / cache drift across the run window.
#
# Usage:
#   scripts/adr034_mtp_paired_bench.sh <gguf> [n_reps] [max_tokens] [prompt]
#
# Outputs a one-line summary per pair plus a final mean + speedup factor.
#
# Why alt-pair: a back-to-back MTP-then-baseline schedule would let thermal
# state from the MTP block dominate the baseline measurement. Alternating
# keeps each mode's runs interleaved across the same thermal window.
#
# Exit status: 0 always (this is a measurement tool, not a gate). The
# F1/F2 ratio thresholds belong in CI elsewhere; this script reports.

# NOTE: deliberately NOT using `set -euo pipefail` — the bench tolerates
# single failed runs (printed as "NaN") rather than terminating the
# alt-pair schedule mid-flight. Per-call errors are visible in stdout.
set -u

GGUF="${1:?usage: $0 <gguf> [n_reps] [max_tokens] [prompt]}"
N_REPS="${2:-3}"
MAX_TOKENS="${3:-128}"
PROMPT="${4:-Write a short essay about the importance of test coverage in software:}"

if [[ ! -f "$GGUF" ]]; then
    echo "fatal: GGUF not found at $GGUF" >&2
    exit 2
fi

HF2Q="${HF2Q:-$(dirname "$0")/../target/release/hf2q}"
if [[ ! -x "$HF2Q" ]]; then
    echo "fatal: hf2q binary not found at $HF2Q — build with 'cargo build --release --bin hf2q' first" >&2
    exit 2
fi

run_one() {
    local mode="$1"  # "spec" or "base"
    local env_val
    case "$mode" in
        spec) env_val=1 ;;
        base) env_val=0 ;;
        *) echo "internal: bad mode $mode" >&2; exit 3 ;;
    esac
    # Capture the trailing tok/s footer line. The qwen35 spec-decode path
    # emits `--- mlx-native (qwen35 spec): N tokens in T.TTs (X.X tok/s, accept Y.Y%) ---`;
    # the greedy path emits `--- mlx-native (qwen35 greedy): N tokens in ... ---`.
    local out
    # HF2Q_QWEN36_AUTOREG=1 is required to load Qwen 3.6 GGUFs (see hf2q
    # generate path's gate at cmd_generate_qwen35 — Wave 5a opt-in).
    # No-op for Qwen 3.5 35B-A3B, so set it unconditionally for the bench.
    # Use `|| true` so a single failed run prints "NaN" rather than tripping
    # `set -e` and terminating the alt-pair schedule.
    out=$(HF2Q_QWEN36_AUTOREG=1 HF2Q_SPEC_DECODE="$env_val" "$HF2Q" generate \
        --model "$GGUF" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0 \
        --no-thinking \
        --ignore-eos 2>&1 || true) || true
    out=$(echo "$out" | tail -3)
    # Extract tok/s and accept rate.
    local tok_s accept
    tok_s=$(echo "$out" | grep -oE '[0-9.]+ tok/s' | tail -1 | sed 's/ tok\/s//')
    accept=$(echo "$out" | grep -oE 'accept [0-9.]+%' | tail -1 | sed 's/accept //')
    echo "${tok_s:-NaN} ${accept:-n/a}"
}

echo "ADR-034 P6 paired MTP bench"
echo "  gguf:        $GGUF"
echo "  hf2q:        $HF2Q"
echo "  reps/mode:   $N_REPS"
echo "  max_tokens:  $MAX_TOKENS"
echo "  prompt:      $(echo "$PROMPT" | head -c 60)..."
echo

declare -a spec_tps base_tps
declare -a spec_accept

for ((i = 1; i <= N_REPS; i++)); do
    read -r tps acc < <(run_one spec)
    spec_tps+=("$tps")
    spec_accept+=("$acc")
    echo "  pair $i  spec: $tps tok/s  (accept=$acc)"

    read -r tps _ < <(run_one base)
    base_tps+=("$tps")
    echo "  pair $i  base: $tps tok/s"
done

# Mean helper using awk (avoid bash float quirks).
mean() {
    local sum=0 n=0
    for v in "$@"; do
        if [[ "$v" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            sum=$(awk -v s="$sum" -v v="$v" 'BEGIN{printf "%.4f", s+v}')
            n=$((n + 1))
        fi
    done
    if [[ "$n" -eq 0 ]]; then echo "NaN"; return; fi
    awk -v s="$sum" -v n="$n" 'BEGIN{printf "%.2f", s/n}'
}

spec_mean=$(mean "${spec_tps[@]}")
base_mean=$(mean "${base_tps[@]}")
accept_mean=$(mean "${spec_accept[@]/\%/}")

speedup=$(awk -v s="$spec_mean" -v b="$base_mean" 'BEGIN{if(b>0){printf "%.2fx", s/b}else{print "NaN"}}')

echo
echo "  spec mean:    $spec_mean tok/s  (accept mean: $accept_mean%)"
echo "  base mean:    $base_mean tok/s"
echo "  speedup:      $speedup  (spec / base)"
