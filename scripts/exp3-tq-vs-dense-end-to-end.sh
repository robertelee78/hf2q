#!/usr/bin/env bash
# EXP-3 — End-to-end TQ-default-on vs dense quality on Gemma-4 APEX-Q5_K_M.
#
# Question the experiment answers: when TQ is the default decode path on a
# non-DWQ model (Gemma-4 APEX-Q5_K_M, our headline README perf model), is the
# user-visible generated text functionally identical to the dense path, or
# does TQ degrade the output?
#
# Method:
#   For each of {short_hello, sourdough, sliding_wrap}:
#     1) capture hf2q TQ-default-on output (no HF2Q_USE_DENSE), N runs
#     2) capture hf2q dense-forced output (HF2Q_USE_DENSE=1), N runs
#     3) compute common_prefix bytes against:
#           - tests/evals/reference/<P>_llama.txt    (llama.cpp baseline)
#           - tests/evals/reference/<P>_hf2q.txt     (frozen hf2q baseline)
#           - the dense run of the same prompt       (TQ-vs-dense self-compare)
#     4) verify determinism: N runs at same mode produce byte-identical output
#
# Acceptance criteria for "TQ tested-to-be-correct, no good enough":
#   - TQ runs must be byte-deterministic (Gate F)
#   - common_prefix(TQ_run, dense_run) must == byte length of shorter run
#     (TQ exactly equals dense on every prompt). ANY divergence = not correct.
#
# If TQ-vs-dense common_prefix == length: TQ is empirically transparent —
# document, ratify default-on, no fix needed.
# If TQ-vs-dense common_prefix < length: TQ degrades; EXP-1 (codec cosine) +
# EXP-2 (FP32 score promotion) are the next steps to find the root cause.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
N_RUNS="${N_RUNS:-3}"
HF2Q_BIN="${HF2Q_BIN:-target/release/hf2q}"
PROMPTS=(short_hello sourdough sliding_wrap)
TS="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="/tmp/exp3-tq-vs-dense-${TS}"
REF_DIR="tests/evals/reference"

[[ -f "$MODEL" ]]    || { echo "error: model not found: $MODEL" >&2; exit 1; }
[[ -x "$HF2Q_BIN" ]] || { echo "error: hf2q binary not at $HF2Q_BIN (run cargo build --release)" >&2; exit 1; }

mkdir -p "$OUT_BASE"
echo "=== EXP-3: TQ-default-on vs dense end-to-end quality ==="
echo "model:      $MODEL"
echo "hf2q HEAD:  $(git rev-parse --short HEAD)"
echo "N_RUNS:     $N_RUNS"
echo "out dir:    $OUT_BASE"
echo

# Capture: $1=mode (tq|dense), $2=run index
capture_one() {
    local mode="$1" run="$2"
    local out_dir="$OUT_BASE/${mode}_run${run}"
    mkdir -p "$out_dir"
    echo "--- capture: mode=$mode run=$run -> $out_dir ---"
    if [[ "$mode" == "dense" ]]; then
        env -u HF2Q_LAYER_POLICY -u HF2Q_TQ_CODEBOOK_BITS HF2Q_USE_DENSE=1 \
            "$HF2Q_BIN" parity capture \
                --model "$MODEL" \
                --output-dir "$out_dir" \
                --prompt all 2>"$out_dir/stderr.log" >"$out_dir/stdout.log"
    else
        # TQ default-on — explicitly unset every dense / policy override so we
        # exercise the SAME decode path a default install would take.
        env -u HF2Q_USE_DENSE -u HF2Q_LAYER_POLICY -u HF2Q_TQ_CODEBOOK_BITS \
            "$HF2Q_BIN" parity capture \
                --model "$MODEL" \
                --output-dir "$out_dir" \
                --prompt all 2>"$out_dir/stderr.log" >"$out_dir/stdout.log"
    fi
}

# Common prefix in bytes between two files. Pure Python — deterministic.
common_prefix() {
    python3 -c '
import sys
a = open(sys.argv[1],"rb").read()
b = open(sys.argv[2],"rb").read()
n = min(len(a),len(b))
i = 0
while i < n and a[i] == b[i]:
    i += 1
print(i)
' "$1" "$2"
}

# Byte length of a file.
byte_len() { wc -c < "$1" | tr -d ' '; }

# 1. Capture both modes, N_RUNS each.
for run in $(seq 1 "$N_RUNS"); do
    capture_one tq    "$run"
    capture_one dense "$run"
done

echo
echo "=== Determinism check (run1 vs runN for each mode) ==="
for mode in tq dense; do
    for prompt in "${PROMPTS[@]}"; do
        f1="$OUT_BASE/${mode}_run1/${prompt}_hf2q.txt"
        all_same=1
        for run in $(seq 2 "$N_RUNS"); do
            fn="$OUT_BASE/${mode}_run${run}/${prompt}_hf2q.txt"
            if ! cmp -s "$f1" "$fn"; then
                all_same=0
                echo "  $mode $prompt run1 vs run$run: DIFFER (cp=$(common_prefix "$f1" "$fn") b)"
            fi
        done
        if (( all_same == 1 )); then
            echo "  $mode $prompt: all $N_RUNS runs byte-identical (Gate F PASS)"
        fi
    done
done

echo
echo "=== Per-prompt comparison table (using run1 of each mode) ==="
printf "%-15s %-9s %-12s %-12s %-12s %-12s %-12s\n" \
    "prompt" "mode" "bytes" "cp_vs_llama" "cp_vs_hfref" "cp_vs_dense" "verdict"
echo "----------------------------------------------------------------------------------------"
for prompt in "${PROMPTS[@]}"; do
    llama="$REF_DIR/${prompt}_llama.txt"
    hfref="$REF_DIR/${prompt}_hf2q.txt"
    tq="$OUT_BASE/tq_run1/${prompt}_hf2q.txt"
    dense="$OUT_BASE/dense_run1/${prompt}_hf2q.txt"

    tq_bytes=$(byte_len "$tq")
    dense_bytes=$(byte_len "$dense")

    cp_tq_llama=$(common_prefix "$tq" "$llama")
    cp_tq_hfref=$(common_prefix "$tq" "$hfref")
    cp_tq_dense=$(common_prefix "$tq" "$dense")

    cp_dense_llama=$(common_prefix "$dense" "$llama")
    cp_dense_hfref=$(common_prefix "$dense" "$hfref")

    # Verdict: TQ output exactly matches dense output (modulo length).
    min_len=$(( tq_bytes < dense_bytes ? tq_bytes : dense_bytes ))
    if [[ "$cp_tq_dense" == "$min_len" ]] && [[ "$tq_bytes" == "$dense_bytes" ]]; then
        verdict="TQ==DENSE"
    elif [[ "$cp_tq_dense" == "$min_len" ]]; then
        verdict="TQ.prefix=DENSE"
    else
        verdict="TQ!=DENSE @${cp_tq_dense}"
    fi

    printf "%-15s %-9s %-12s %-12s %-12s %-12s %-12s\n" \
        "$prompt" "TQ"    "$tq_bytes"    "$cp_tq_llama"    "$cp_tq_hfref"    "$cp_tq_dense" "$verdict"
    printf "%-15s %-9s %-12s %-12s %-12s %-12s %-12s\n" \
        "$prompt" "DENSE" "$dense_bytes" "$cp_dense_llama" "$cp_dense_hfref" "(self)" ""
done
echo
echo "Captures saved under: $OUT_BASE"
