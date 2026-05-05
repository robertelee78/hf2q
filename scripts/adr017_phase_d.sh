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
#   * kv_persist_phase_d_r_p1_concurrent_eviction_e2e — K2 polish
#     (iter-12) closing the iter-8 honest caveat. Fires the eviction
#     trick from a sibling thread ~100ms INTO a decode (concurrent
#     with in-flight inference) rather than between decodes. Asserts
#     full-decode-wall-time overhead <= 5%. OFF by default; set
#     HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=1 to opt in. Distinct
#     from HF2Q_KV_PERSIST_PHASE_D_R_P1 so operators can run the two
#     K2 measurements independently.
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
#   scripts/adr017_phase_d.sh --rp1                    # opt in K2 R-P1 sustained-decode overhead
#   scripts/adr017_phase_d.sh --rp1-concurrent         # opt in K2 R-P1 concurrent-eviction polish
#   scripts/adr017_phase_d.sh --rp5                    # opt in R-P5 cold-process resume ship-gate
#   scripts/adr017_phase_d.sh --rp6                    # opt in R-P6 4-agent shared-prefix ship-gate
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
ENABLE_STRESS=0
STRESS_DURATION=1800
STRESS_BUDGET_MB=4096

usage() {
  cat <<EOF
Usage: scripts/adr017_phase_d.sh [--model PATH] [--prefill N] [--peer]
                                 [--rp1] [--rp1-concurrent] [--rp5] [--rp6]
                                 [--stress] [--stress-duration SEC] [--stress-budget-mb MB]
                                 [--skip-process-audit]

Drives ADR-017 Phase D coherence + perf validation tests.

  --model PATH             GGUF path (default: $DEFAULT_MODEL_PATH)
  --prefill N              R-P4 prefill length in tokens (default: $DEFAULT_PREFILL_LEN)
  --peer                   Enable llama-completion R-C4 peer arm
  --rp1                    Opt in K2 R-P1 sustained-decode overhead test
                           (sets HF2Q_KV_PERSIST_PHASE_D_R_P1=1)
  --rp1-concurrent         Opt in K2 R-P1 polish: concurrent-eviction-during-decode
                           (sets HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=1)
  --rp5                    Opt in R-P5 cold-process resume ship-gate
                           (sets HF2Q_KV_PERSIST_PHASE_D_R_P5=1)
  --rp6                    Opt in R-P6 4-agent shared-prefix ship-gate
                           (sets HF2Q_KV_PERSIST_PHASE_D_R_P6=1)
  --stress                 Run the kv_persist_stress 24h smoke test alongside
                           the gemma4_roundtrip suite. Default duration 1800s
                           (30 min); pass --stress-duration to override (full
                           24h gate uses 86400s, validates iter-to-iter RSS
                           leak ≤ 5% across hundreds of swap cycles).
  --stress-duration SEC    Override stress duration in seconds (default 1800)
  --stress-budget-mb MB    Override stress on-disk cache budget in MB (default 4096
                           = §R-F5 spec). Threaded through to the spawned
                           server's HF2Q_KV_PERSIST_BUDGET_BYTES so the
                           writer's post-write LRU eviction has a budget to
                           enforce.
  --skip-process-audit     Bypass pre-bench process audit (NOT recommended)
  -h, --help               Show this help

Default-on tests (always run):
  R-C4 internal coherence (sourdough byte-equality)
  R-P4 ship-gate (cache_hit_TTFT(32K) / no_cache_TTFT(32K) <= 0.20)

Opt-in tests must be explicitly enabled via the flags above (or by exporting
the corresponding HF2Q_KV_PERSIST_PHASE_D_* / HF2Q_KV_PERSIST_STRESS_24H env
var to 1 before invocation).

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
    --rp1) export HF2Q_KV_PERSIST_PHASE_D_R_P1=1; shift ;;
    --rp1-concurrent) export HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=1; shift ;;
    --rp5) export HF2Q_KV_PERSIST_PHASE_D_R_P5=1; shift ;;
    --rp6) export HF2Q_KV_PERSIST_PHASE_D_R_P6=1; shift ;;
    --stress) ENABLE_STRESS=1; shift ;;
    --stress-duration)
      [[ $# -ge 2 ]] || { echo "error: --stress-duration requires an argument" >&2; exit 1; }
      STRESS_DURATION="$2"; shift 2 ;;
    --stress-budget-mb)
      [[ $# -ge 2 ]] || { echo "error: --stress-budget-mb requires an argument" >&2; exit 1; }
      STRESS_BUDGET_MB="$2"; shift 2 ;;
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
echo "    HF2Q_KV_PERSIST_PHASE_D_R_P1=${HF2Q_KV_PERSIST_PHASE_D_R_P1:-0}  (K2 ship-gate; OFF by default; set =1 in env or pass --rp1 to opt in)"
echo "    HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=${HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT:-0}  (K2 polish; concurrent-eviction-during-decode; OFF by default; set =1 in env or pass --rp1-concurrent to opt in)"
echo "    HF2Q_KV_PERSIST_PHASE_D_R_P5=${HF2Q_KV_PERSIST_PHASE_D_R_P5:-0}  (R-P5 cold-process resume ship-gate; OFF by default; set =1 in env or pass --rp5 to opt in)"
echo "    HF2Q_KV_PERSIST_PHASE_D_R_P6=${HF2Q_KV_PERSIST_PHASE_D_R_P6:-0}  (R-P6 4-agent shared-prefix ship-gate; OFF by default; set =1 in env or pass --rp6 to opt in)"
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
    HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT="${HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT:-0}" \
    HF2Q_KV_PERSIST_PHASE_D_R_P5="${HF2Q_KV_PERSIST_PHASE_D_R_P5:-0}" \
    HF2Q_KV_PERSIST_PHASE_D_R_P6="${HF2Q_KV_PERSIST_PHASE_D_R_P6:-0}" \
    cargo test --release --test kv_persist_gemma4_roundtrip \
      -- --test-threads=1 --nocapture; then
  echo
  echo "=== ADR-017 Phase D — Gate accounting ==="
  echo "  R-C4 internal: ALWAYS RUN (PASS implied by exit 0)"
  echo "  R-P4         : ALWAYS RUN (PASS implied by exit 0)"
  echo "  R-C4 peer    : $([ "$ENABLE_PEER" -eq 1 ] && echo 'OPT-IN RUN' || echo 'SKIPPED (--peer not set)')"
  echo "  K2 R-P1      : $([ "${HF2Q_KV_PERSIST_PHASE_D_R_P1:-0}" = 1 ] && echo 'OPT-IN RUN' || echo 'SKIPPED (HF2Q_KV_PERSIST_PHASE_D_R_P1 unset)')"
  echo "  K2 R-P1 conc : $([ "${HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT:-0}" = 1 ] && echo 'OPT-IN RUN' || echo 'SKIPPED (HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT unset)')"
  echo "  R-P5         : $([ "${HF2Q_KV_PERSIST_PHASE_D_R_P5:-0}" = 1 ] && echo 'OPT-IN RUN' || echo 'SKIPPED (HF2Q_KV_PERSIST_PHASE_D_R_P5 unset)')"
  echo "  R-P6         : $([ "${HF2Q_KV_PERSIST_PHASE_D_R_P6:-0}" = 1 ] && echo 'OPT-IN RUN' || echo 'SKIPPED (HF2Q_KV_PERSIST_PHASE_D_R_P6 unset)')"
  echo
  echo "Run scripts/adr017_phase_d.sh --help for env-var opt-ins to exercise additional gates."
  EXIT_CODE=0
else
  rc=$?
  echo
  echo "=== ADR-017 Phase D — TEST FAILURE (cargo test exit=$rc) ==="
  EXIT_CODE=2
fi

# 5. Optional stress 24h gate (separate test crate; long-running).
if [[ "$ENABLE_STRESS" -eq 1 ]] && [[ "$EXIT_CODE" -eq 0 ]]; then
  echo
  echo "--- cargo test --release --test kv_persist_stress (24h smoke gate) ---"
  echo "    HF2Q_KV_PERSIST_STRESS_24H=1"
  echo "    HF2Q_KV_PERSIST_E2E_MODEL_PATH=$MODEL_PATH"
  echo "    HF2Q_KV_PERSIST_STRESS_DURATION_SEC=$STRESS_DURATION"
  echo "    HF2Q_KV_PERSIST_STRESS_BUDGET_MB=$STRESS_BUDGET_MB"
  echo "    HF2Q_KV_PERSIST_STRESS_MAX_L=4096"
  echo
  echo "    NB: spec is 86400s (24h). 1800s default = smoke validating"
  echo "    the iter-10/11 leak fix + R-F5 budget eviction at SPEC config."
  echo "    The post-warmup baseline + 5% rss_tol + 4GB cache budget are"
  echo "    the spec values; the test FAILs if RSS or cache exceed."
  echo

  if HF2Q_KV_PERSIST_STRESS_24H=1 \
      HF2Q_KV_PERSIST_E2E_MODEL_PATH="$MODEL_PATH" \
      HF2Q_KV_PERSIST_STRESS_DURATION_SEC="$STRESS_DURATION" \
      HF2Q_KV_PERSIST_STRESS_BUDGET_MB="$STRESS_BUDGET_MB" \
      HF2Q_KV_PERSIST_STRESS_MAX_L="${HF2Q_KV_PERSIST_STRESS_MAX_L:-4096}" \
      cargo test --release --test kv_persist_stress kv_persist_stress_24h \
        -- --test-threads=1 --nocapture; then
    echo
    echo "=== ADR-017 Phase D — Stress 24h gate ==="
    echo "  Stress 24h     : OPT-IN RUN (PASS implied by exit 0)"
  else
    rc=$?
    echo
    echo "=== ADR-017 Phase D — STRESS TEST FAILURE (cargo test exit=$rc) ==="
    EXIT_CODE=2
  fi
fi

exit "$EXIT_CODE"
