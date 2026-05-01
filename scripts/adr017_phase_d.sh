#!/usr/bin/env bash
# adr017_phase_d.sh — ADR-017 Phase D coherence + perf validation operator recipe.
#
# Drives the env-gated Phase D tests in tests/kv_persist_gemma4_roundtrip.rs:
#   * kv_persist_gemma4_roundtrip_matrix_e2e — full 60-cell sweep
#     (production-quant subset = 36 runnable cells)
#   * kv_persist_phase_d_coherence_e2e — sourdough byte-exact never-evicted
#     == evicted+restored (R-C4 internal); optional peer arm asserts
#     >=3094 bytes shared with llama-completion
#   * kv_persist_phase_d_r_p4_e2e — at L=32K, cache_hit_ttft /
#     no_cache_ttft <= 0.20 ship-gate
#   * kv_persist_phase_d_r_p1_decode_overhead_e2e — K2 ship-gate
#     (LAST outstanding ADR-017 kill-gate); 5-baseline + 5-sustained
#     decode overhead under sustained eviction; <= 5%. OFF by default;
#     set HF2Q_KV_PERSIST_PHASE_D_R_P1=1 (env or via this script's
#     wiring below) to opt in.
#
# Pre-conditions (per ADR-017 §Phase D + feedback_bench_process_audit):
#   - Cold M5 Max (~1-min idle since previous run; pmset -g thermlog clean)
#   - mcp-brain-server SIGSTOP'd (or absent) — pre-bench audit fails
#     hard if any of mcp-brain-server / llama-server / ollama is running
#   - cargo build --release succeeded with the hf2q binary at
#     target/release/hf2q
#   - GGUF fixture present at ADR017_PHASE_D_MODEL_PATH (default:
#     the canonical Gemma 4 26B Q4_0 abliterated path).
#
# Usage:
#   scripts/adr017_phase_d.sh                          # default model + 32K cell + no peer arm
#   scripts/adr017_phase_d.sh --peer                   # enable llama-completion R-C4 peer arm
#   scripts/adr017_phase_d.sh --prefill 32768          # explicit R-P4 prefill length
#   scripts/adr017_phase_d.sh --model /path/to.gguf    # override GGUF
#   scripts/adr017_phase_d.sh --skip-process-audit     # bypass pre-bench audit (NOT recommended)
#
# Exit codes:
#   0  all gates PASS
#   1  usage / env error
#   2  cargo test exited non-zero (gate FAIL or test crash)
#   3  prerequisites missing (binary not built, model not found, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL_PATH="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
DEFAULT_PREFILL_LEN="32768"

MODEL_PATH="${ADR017_PHASE_D_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
PREFILL_LEN="$DEFAULT_PREFILL_LEN"
ENABLE_PEER=0
SKIP_AUDIT=0

usage() {
  cat <<EOF
Usage: scripts/adr017_phase_d.sh [--model PATH] [--prefill N] [--peer] [--skip-process-audit]

Drives ADR-017 Phase D coherence + perf validation tests.

  --model PATH             GGUF path (default: $DEFAULT_MODEL_PATH)
  --prefill N              R-P4 prefill length in tokens (default: $DEFAULT_PREFILL_LEN)
  --peer                   Enable llama-completion R-C4 peer arm
  --skip-process-audit     Bypass pre-bench process audit (NOT recommended)
  -h, --help               Show this help

Exit codes: 0=PASS  1=usage  2=test-fail  3=prereq-missing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "error: --model requires an argument" >&2; exit 1; }
      MODEL_PATH="$2"; shift 2 ;;
    --prefill)
      [[ $# -ge 2 ]] || { echo "error: --prefill requires an argument" >&2; exit 1; }
      PREFILL_LEN="$2"; shift 2 ;;
    --peer) ENABLE_PEER=1; shift ;;
    --skip-process-audit) SKIP_AUDIT=1; shift ;;
    -*) usage >&2; echo "error: unknown flag: $1" >&2; exit 1 ;;
    *)  usage >&2; echo "error: unexpected positional arg: $1" >&2; exit 1 ;;
  esac
done

HF2Q_BIN="$REPO_ROOT/target/release/hf2q"

echo "=== ADR-017 Phase D — coherence + perf validation ==="
echo "REPO_ROOT:   $REPO_ROOT"
echo "hf2q bin:    $HF2Q_BIN"
echo "model:       $MODEL_PATH"
echo "prefill:     $PREFILL_LEN tokens"
echo "peer arm:    $([ "$ENABLE_PEER" -eq 1 ] && echo enabled || echo disabled)"
echo

# 1. Prerequisite checks.
if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "error: hf2q binary not found at $HF2Q_BIN" >&2
  echo "       run: cargo build --release" >&2
  exit 3
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: GGUF not found at $MODEL_PATH" >&2
  echo "       set ADR017_PHASE_D_MODEL_PATH or pass --model PATH" >&2
  exit 3
fi

# 2. Pre-bench process audit (per feedback_bench_process_audit). Refuses
#    to run if mcp-brain-server / llama-server / ollama is detected; the
#    M5 Max iter-4..iter-8c contamination episode showed +1.5 t/s noise
#    purely from these processes contending for the SoC.
if [[ "$SKIP_AUDIT" -eq 0 ]]; then
  echo "--- Pre-bench process audit ---"
  PS_OUT="$(ps -Ao comm,pid,%cpu 2>/dev/null || true)"
  if [[ -z "$PS_OUT" ]]; then
    echo "warning: ps unavailable; cannot audit. Override with --skip-process-audit if you accept the risk." >&2
    exit 3
  fi
  CONTAMINANTS=""
  while IFS= read -r line; do
    case "$line" in
      *mcp-brain-server*|*llama-server*|*llama-cli*|*ollama*)
        CONTAMINANTS="${CONTAMINANTS}${line}\n"
        ;;
    esac
  done <<<"$PS_OUT"
  if [[ -n "$CONTAMINANTS" ]]; then
    echo "error: competing processes detected — measurement INVALID:" >&2
    printf '%b' "$CONTAMINANTS" >&2
    echo "       SIGSTOP these or stop them, then rerun." >&2
    echo "       Override with --skip-process-audit if you accept the risk." >&2
    exit 3
  fi
  echo "OK — clean SoC."
else
  echo "--- Pre-bench process audit SKIPPED (--skip-process-audit) ---"
  echo "warning: results may be contaminated by competing processes" >&2
fi

# 3. RAM check (per feedback_check_ram_before_inference).
if command -v vm_stat >/dev/null 2>&1; then
  echo "--- RAM check ---"
  vm_stat | head -5
  echo
fi

# 4. Run Phase D tests with the env vars wired up. The matrix runner
#    + Phase D tests all gate on env; cargo test always succeeds when
#    env is unset. Setting both gates here ensures all 3 master tests
#    fire with measurement: matrix sweep + coherence + R-P4 ship-gate.
echo "--- cargo test --release --test kv_persist_gemma4_roundtrip ---"
echo "    HF2Q_KV_PERSIST_E2E=1"
echo "    HF2Q_KV_PERSIST_PHASE_D=1"
echo "    HF2Q_USE_DENSE=1"
echo "    HF2Q_KV_PERSIST_E2E_MODEL_PATH=$MODEL_PATH"
echo "    HF2Q_KV_PERSIST_E2E_PREFILL_LEN=$PREFILL_LEN"
echo "    HF2Q_KV_PERSIST_PHASE_D_PEER=$([ "$ENABLE_PEER" -eq 1 ] && echo 1 || echo 0)"
echo "    HF2Q_KV_PERSIST_PHASE_D_R_P1=${HF2Q_KV_PERSIST_PHASE_D_R_P1:-0}  (K2 ship-gate; OFF by default; set =1 in env to opt in)"
echo

cd "$REPO_ROOT"

EXIT_CODE=0
if HF2Q_KV_PERSIST_E2E=1 \
    HF2Q_KV_PERSIST_PHASE_D=1 \
    HF2Q_USE_DENSE=1 \
    HF2Q_KV_PERSIST_E2E_MODEL_PATH="$MODEL_PATH" \
    HF2Q_KV_PERSIST_E2E_PREFILL_LEN="$PREFILL_LEN" \
    HF2Q_KV_PERSIST_PHASE_D_PEER="$([ "$ENABLE_PEER" -eq 1 ] && echo 1 || echo 0)" \
    HF2Q_KV_PERSIST_PHASE_D_R_P1="${HF2Q_KV_PERSIST_PHASE_D_R_P1:-0}" \
    cargo test --release --test kv_persist_gemma4_roundtrip \
      -- --test-threads=1 --nocapture; then
  echo
  echo "=== ADR-017 Phase D — ALL GATES PASS ==="
  EXIT_CODE=0
else
  rc=$?
  echo
  echo "=== ADR-017 Phase D — TEST FAILURE (cargo test exit=$rc) ==="
  EXIT_CODE=2
fi

exit "$EXIT_CODE"
