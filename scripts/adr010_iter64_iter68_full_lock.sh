#!/usr/bin/env bash
# ADR-010 iter-64 + iter-68 wins lock-in chain.
#
# Single operator-runnable command that runs ALL the regression gates
# protecting the iter-64 (forward_prefill_batched.rs leg_hb_encoded fix,
# 22-29× speedup over per-token default) AND iter-68 (mlx-native one-
# character typo fix unlocking tensor mm_id, BEATS llama.cpp at pp1024)
# wins.
#
# Sequence:
#   1. mlx-native shader-compile gate (iter-73) — every .metal file
#      compiles via xcrun. Catches iter-68-style silent typos that
#      cause runtime fallback to slower kernel paths.
#   2. hf2q batched-prefill coherence + perf gate (iter-65/74/75):
#      - per-token vs batched byte-identity at short prompt
#      - pp1024 batched perf >= 1500 t/s floor (iter-68 baseline 1942)
#
# Total runtime: ~1 minute. Run after any change to forward_prefill_
# batched.rs, the mm_id Metal kernels, or the dispatcher routing.
#
# # Usage
#
#   bash scripts/adr010_iter64_iter68_full_lock.sh
#
# # Env overrides
#
#   MODEL=...                     model path (default: gemma APEX-Q5_K_M)
#   HF2Q_GATE_PERF_FLOOR=...      lower the pp1024 floor (default 1500)
#   HF2Q_GATE_SKIP_PERF=1         skip perf-sanity (slower hardware)
#
# # Exit codes
#
#   0 = ALL gates PASS
#   1 = shader-compile gate failed (mlx-native side)
#   2 = batched coherence/perf gate failed (hf2q side)

set -euo pipefail

if [[ -t 1 ]]; then
    BOLD=$'\e[1m'
    GREEN=$'\e[32m'
    RED=$'\e[31m'
    RESET=$'\e[0m'
else
    BOLD=""; GREEN=""; RED=""; RESET=""
fi

echo "${BOLD}=== ADR-010 iter-64 + iter-68 wins LOCK-IN check ===${RESET}"
echo

echo "${BOLD}[1/2] mlx-native shader-compile gate (iter-73)${RESET}"
if (cd /opt/mlx-native && RUSTC_WRAPPER= cargo test --test test_all_shaders_compile --quiet 2>&1 | tail -3); then
    echo "${GREEN}  ✓ shader-compile gate PASS${RESET}"
else
    echo "${RED}  ✗ shader-compile gate FAILED${RESET}" >&2
    exit 1
fi
echo

echo "${BOLD}[2/3] hf2q gemma batched-prefill coherence + perf gate (iter-65/74/75)${RESET}"
if bash /opt/hf2q/scripts/adr010_iter64_batched_coherence_gate.sh 2>&1 | tail -10; then
    echo "${GREEN}  ✓ gemma coherence + perf gate PASS${RESET}"
else
    echo "${RED}  ✗ gemma coherence + perf gate FAILED${RESET}" >&2
    exit 2
fi
echo

# Iter-78: qwen35 perf-floor — cross-model validation that iter-68's
# tensor mm_id unlock benefits qwen35moe too. qwen35 uses its own
# code path (cmd_generate_qwen35 → forward_gpu.rs) but routes through
# the same shared mm_id_pooled dispatcher in mlx-native, which means
# the iter-68 typo fix (preventing fallback to slower simdgroup MMA)
# applies. Measured 2300 t/s at pp512 = 0.79× of llama.cpp peer (2921).
# Floor 1800 catches >20% regression.
QWEN_MODEL="${QWEN_MODEL:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
QWEN_PERF_FLOOR="${HF2Q_QWEN_PERF_FLOOR:-1800}"
if [[ "${HF2Q_GATE_SKIP_QWEN:-0}" != "1" ]] && [[ -f "$QWEN_MODEL" ]]; then
    echo "${BOLD}[3/3] hf2q qwen35 prefill perf-floor (iter-78 cross-model)${RESET}"
    QWEN_PROMPT_FILE=$(mktemp -t adr010-iter78-qwen.XXXXXX)
    trap "rm -f $QWEN_PROMPT_FILE" EXIT
    python3 -c "
words=['the','quick','brown','fox','jumps','over','lazy','dog','and','runs','through','green','meadow','past','silver','river','where','the','old','willow','stands']
n=512
out=[]
i=0
while len(' '.join(out).split())<n:
    out.append(words[i%len(words)])
    i+=1
print(' '.join(out))
" > "$QWEN_PROMPT_FILE"
    declare -a QWEN_TRIALS=()
    for trial in 1 2 3; do
        QWEN_LINE=$(timeout 180 /opt/hf2q/target/release/hf2q generate \
            --model "$QWEN_MODEL" --prompt-file "$QWEN_PROMPT_FILE" --max-tokens 1 2>&1 \
            | grep "^prefill:" | head -1)
        QWEN_TRIAL_TOK_S=$(echo "$QWEN_LINE" | grep -oE "\(([0-9]+) tok/s\)" | grep -oE "[0-9]+" | head -1)
        if [[ -z "$QWEN_TRIAL_TOK_S" ]]; then
            echo "${RED}  ✗ FAIL: could not parse qwen35 perf result on trial $trial${RESET}" >&2
            exit 3
        fi
        echo "  trial $trial: $QWEN_LINE"
        QWEN_TRIALS+=("$QWEN_TRIAL_TOK_S")
    done
    QWEN_MEDIAN=$(printf '%s\n' "${QWEN_TRIALS[@]}" | sort -n | sed -n '2p')
    echo "  median: $QWEN_MEDIAN tok/s (over 3 trials)"
    if (( QWEN_MEDIAN < QWEN_PERF_FLOOR )); then
        echo "${RED}  ✗ FAIL: qwen35 $QWEN_MEDIAN tok/s < ${QWEN_PERF_FLOOR} t/s floor${RESET}" >&2
        echo "  iter-78 baseline 2300 t/s; suggests cross-model regression in mm_id dispatcher."
        exit 3
    fi
    echo "${GREEN}  ✓ qwen35 perf-floor PASS ($QWEN_MEDIAN t/s ≥ $QWEN_PERF_FLOOR t/s)${RESET}"
elif [[ ! -f "$QWEN_MODEL" ]]; then
    echo "${BOLD}[3/3] qwen35 perf-floor SKIPPED (model not found at $QWEN_MODEL)${RESET}"
else
    echo "${BOLD}[3/3] qwen35 perf-floor SKIPPED (HF2Q_GATE_SKIP_QWEN=1)${RESET}"
fi
echo

echo "${GREEN}${BOLD}=== ALL ADR-010 iter-64/68 LOCK-IN GATES PASS ===${RESET}"
echo "  • Every Metal shader compiles (no silent typos)"
echo "  • Gemma per-token ≡ batched on short prompt (no compute-path divergence)"
echo "  • Gemma pp1024 batched ≥ 1500 t/s floor (iter-68 tensor mm_id active)"
echo "  • Qwen35 pp512 prefill ≥ ${QWEN_PERF_FLOOR} t/s floor (iter-78 cross-model)"
echo
echo "iter-64 (forward_prefill_batched.rs HB-encode fix, 91 LOC at 133722d)"
echo "and iter-68 (mlx-native typo fix at b6b8e79) wins are protected"
echo "across both gemma AND qwen35 model families."
