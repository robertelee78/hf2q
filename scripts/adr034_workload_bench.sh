#!/usr/bin/env bash
# ADR-034 workload-aware speculative-decode bench.
#
# Captures the workload-sensitivity finding from
# /opt/hf2q/docs/ADR-034-speculative-decode-end-to-end.md §Mission status
# (commits 4dd6df2a + 00b9ac54). Runs 3-rep paired benches across:
#
#   - Base               (HF2Q_SPEC_DECODE=0)
#   - MTP K=1 greedy    (HF2Q_SPEC_DECODE=1 --temperature 0)
#   - MTP K=1 MH t=0.5  (HF2Q_SPEC_DECODE=1 --temperature 0.5)
#   - DFlash BS=2       (HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=2)
#   - DFlash BS=4       (peak per ADR §Mission status, both workloads)
#
# Across two prompt classes:
#   - Essay  (creative, high-entropy)
#   - Code-gen (deterministic, low-entropy)
#
# Outputs a table summarising mean + tok/s + accept-rate per config.
#
# Usage:
#   scripts/adr034_workload_bench.sh [gguf_path]
#
# Default GGUF: models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0-mtp.gguf
#
# This is a measurement tool — exit status is always 0; the table is
# the contract. Use for regression detection after kernel changes.
#
# Per-config 3-rep alt-paired against base interleaves runs to control
# for thermal/cache drift (matches the methodology of
# scripts/adr034_mtp_paired_bench.sh).

set -u

GGUF="${1:-models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0-mtp.gguf}"
MAX_TOKENS="${MAX_TOKENS:-128}"
N_REPS="${N_REPS:-3}"
HF2Q="${HF2Q:-./target/release/hf2q}"

if [[ ! -f "$GGUF" ]]; then
    echo "fatal: GGUF not found at $GGUF" >&2
    exit 2
fi
if [[ ! -x "$HF2Q" ]]; then
    echo "fatal: hf2q binary not found at $HF2Q — build with 'cargo build --release --bin hf2q' first" >&2
    exit 2
fi

# Defensive: kill any stale hf2q generate processes from previous bench
# runs that may have been left running (e.g., timeout-orphaned wrappers,
# background bench runs killed mid-flight, etc.). Without this, stale
# processes compete for GPU + cache space and produce corrupted bench
# results — empirically observed at HEAD `b40db9a8` 2026-05-22 when a
# 37-minute-old stale process from a prior iter contended for GPU and
# dropped base throughput from 22 t/s → 2.8 t/s.
#
# Only matches `hf2q generate` (not the harness/build/test processes),
# so won't kill the user's own development workflow.
if pgrep -f "hf2q generate" > /dev/null 2>&1; then
    echo "warn: killing stale 'hf2q generate' processes from prior runs" >&2
    pkill -f "hf2q generate" 2>/dev/null || true
    sleep 3
fi

PROMPT_ESSAY="Write a short essay about the importance of test coverage in software:"
PROMPT_CODEGEN="Write a Python function that computes the Fibonacci sequence iteratively:"

# Run one config N_REPS times, print mean tok/s + median accept rate.
# Args: $1=label, $2=prompt, $3=extra env vars, $4=cli args
run_config() {
    local label="$1"
    local prompt="$2"
    local env_vars="$3"
    local cli_args="$4"

    local tps_sum="0"
    local acc=""
    local samples=()
    for i in $(seq 1 "$N_REPS"); do
        local out
        # Use `env` so the env_vars string (e.g., "HF2Q_SPEC_DECODE=0")
        # is interpreted as variable assignments rather than command words.
        # Wrapping in bash -c lets us combine `env` setup + invocation
        # without bash's pre-command-word env-assignment limitation.
        # shellcheck disable=SC2086
        out=$(env HF2Q_QWEN36_AUTOREG=1 $env_vars timeout 240 "$HF2Q" generate \
            --model "$GGUF" \
            --prompt "$prompt" \
            --max-tokens "$MAX_TOKENS" \
            $cli_args \
            --no-thinking --ignore-eos 2>&1 || true) || true
        local tps
        tps=$(echo "$out" | grep -oE '[0-9.]+ tok/s' | tail -1 | sed 's/ tok\/s//')
        local this_acc
        this_acc=$(echo "$out" | grep -oE 'accept [0-9.]+%' | tail -1 | sed 's/accept //')
        samples+=("${tps:-NaN}")
        # Use the last seen accept rate (deterministic at greedy/sampler-seeded paths)
        if [[ -n "$this_acc" ]]; then acc="$this_acc"; fi
    done
    # Compute mean via awk.
    local mean
    mean=$(printf '%s\n' "${samples[@]}" | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "NaN"}')
    printf '  %-32s | reps=%s mean=%s tok/s | accept=%s\n' "$label" "${samples[*]}" "$mean" "${acc:-n/a}"
}

echo "ADR-034 workload-aware speculative-decode bench"
echo "  gguf:         $GGUF"
echo "  hf2q:         $HF2Q"
echo "  reps/config:  $N_REPS"
echo "  max_tokens:   $MAX_TOKENS"
echo

# ── Warmup pass ──
#
# Run a 1-rep throwaway invocation of EACH config-class (base + spec
# + dflash) at small max_tokens=16 to populate:
#   - macOS unified-memory page cache for the GGUF + DFlash drafter
#   - Metal pipeline cache (PSO JIT compile for all kernels touched)
#   - GPU thermal warmup
#
# Without this, the first config of the actual measurement run shows
# cold-cache numbers 15-25% below steady-state — empirically observed
# at HEAD ce3d32e6 (commit 6e94724d documents the +19-24% post-task-#95
# bench discrepancy this caused before warmup was added).
#
# Captured into /dev/null; no output. ~30s total at small max_tokens.
echo "Warmup pass (populates page cache + Metal pipelines + GPU thermal)..."
(
    HF2Q_QWEN36_AUTOREG=1 HF2Q_SPEC_DECODE=0 \
        "$HF2Q" generate --model "$GGUF" --prompt "warmup" \
        --max-tokens 16 --temperature 0 --no-thinking --ignore-eos \
        > /dev/null 2>&1 || true
    HF2Q_QWEN36_AUTOREG=1 HF2Q_SPEC_DECODE=1 \
        "$HF2Q" generate --model "$GGUF" --prompt "warmup" \
        --max-tokens 16 --temperature 0 --no-thinking --ignore-eos \
        > /dev/null 2>&1 || true
    HF2Q_QWEN36_AUTOREG=1 HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=4 \
        "$HF2Q" generate --model "$GGUF" --prompt "warmup" \
        --max-tokens 16 --temperature 0 --no-thinking --ignore-eos \
        > /dev/null 2>&1 || true
)
echo "  done"
echo

echo "=== Essay prompt (creative / high-entropy) ==="
echo "  Prompt: '$PROMPT_ESSAY'"
echo
run_config "Base"                  "$PROMPT_ESSAY" "HF2Q_SPEC_DECODE=0"  "--temperature 0"
run_config "MTP K=1 greedy"        "$PROMPT_ESSAY" "HF2Q_SPEC_DECODE=1"  "--temperature 0"
run_config "MTP K=1 MH temp=0.5"   "$PROMPT_ESSAY" "HF2Q_SPEC_DECODE=1"  "--temperature 0.5"
run_config "DFlash BS=2"           "$PROMPT_ESSAY" "HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=2"  "--temperature 0"
run_config "DFlash BS=4"           "$PROMPT_ESSAY" "HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=4"  "--temperature 0"
echo

echo "=== Code-gen prompt (deterministic / low-entropy) ==="
echo "  Prompt: '$PROMPT_CODEGEN'"
echo
run_config "Base"                  "$PROMPT_CODEGEN" "HF2Q_SPEC_DECODE=0"  "--temperature 0"
run_config "MTP K=1 greedy"        "$PROMPT_CODEGEN" "HF2Q_SPEC_DECODE=1"  "--temperature 0"
run_config "MTP K=1 MH temp=0.5"   "$PROMPT_CODEGEN" "HF2Q_SPEC_DECODE=1"  "--temperature 0.5"
run_config "DFlash BS=2"           "$PROMPT_CODEGEN" "HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=2"  "--temperature 0"
run_config "DFlash BS=4"           "$PROMPT_CODEGEN" "HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=4"  "--temperature 0"
echo

echo "Reference baselines (Qwen 3.6 27B Q8_0):"
echo "  Pre task #95 (HEAD 00b9ac54):"
echo "    Essay:    Base 21.9, MTP greedy 26.2 @ 68%, MTP MH 27.5 @ 78%, DFlash BS=2 16.5, BS=4 16.9"
echo "    Code-gen: Base 22.0, MTP greedy 29.9 @ 91%, MTP MH 28.6 @ 84%, DFlash BS=2 16.8, BS=4 18.4"
echo "  Post task #95 sub-iters A-G+E (HEAD ce3d32e6 2026-05-22, warmup-corrected):"
echo "    Essay:    Base 21.9, MTP greedy 26.5 @ 68%, MTP MH 27.4 @ 78%, DFlash BS=2 19.9, BS=4 20.2"
echo "    Code-gen: Base 22.0, MTP greedy 29.9 @ 91%, MTP MH 28.6 @ 84%, DFlash BS=2 20.8, BS=4 22.4"
echo "  Post task #95 sub-iter H (HEAD 1f130770 2026-05-22, warmup-corrected):"
echo "    Essay:    Base 21.9, MTP greedy 26.4 @ 68%, MTP MH 27.5 @ 78%, DFlash BS=2 20.7, BS=4 21.0"
echo "    Code-gen: Base 22.0, MTP greedy 30.0 @ 91%, MTP MH 28.6 @ 84%, DFlash BS=2 21.6, BS=4 23.2"
echo "  Post K=N CORRECTION + ADR scrub (HEAD e044131d 2026-05-22, warmup-corrected, stability validation):"
echo "    Essay:    Base 21.9, MTP greedy 26.6 @ 68%, MTP MH 27.6 @ 78%, DFlash BS=2 20.8, BS=4 21.0"
echo "    Code-gen: Base 22.0, MTP greedy 30.0 @ 91%, MTP MH 28.7 @ 84%, DFlash BS=2 21.6, BS=4 23.1"
echo "    State stable vs sub-iter H (±0.2 t/s noise band). No regression from task #96 revert + ADR correction."
echo "  Post 10-iter doc-scrub + pkill defense (HEAD eeaa51be 2026-05-22, focused code-gen 3-rep):"
echo "    Code-gen: Base 21.93, MTP greedy 29.93 @ 91% (1.36x base), MTP MH 28.77 @ 84%, DFlash BS=2 21.63, BS=4 23.17 (0.77x of MTP greedy)"
echo "    All 5 configs match documented values within ±0.1 t/s noise band. Production state empirically stable across 10-iter cumulative doc-scrub period (45+ stale claims corrected, zero code-side drift)."
echo
echo "Production recommendation:"
echo "  Code-gen / deterministic: HF2Q_SPEC_DECODE=1 --temperature 0     (1.36x base, MTP K=1 greedy wins)"
echo "  Essay / creative:         HF2Q_SPEC_DECODE=1 --temperature 0.5   (1.26x base, MTP K=1 MH wins)"
echo "  DFlash opt-in:            HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_BLOCK_SIZE=4  (0.75x of MTP greedy on code-gen post-task-#95; research-quality but closer to viable)"
