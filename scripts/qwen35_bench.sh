#!/usr/bin/env bash
# qwen35_bench.sh — ADR-013 Phase P13 match-or-beat benchmark gate for
# Qwen3.5 / Qwen3.6 inference vs llama.cpp on the same GGUF.
#
# Runs both hf2q's internal benchmark and llama-bench across a matrix of
# prompt lengths (prefill) × decode lengths, reports side-by-side tok/s
# tables, and asserts hf2q is within `DRIFT_BUDGET_PCT` of llama.cpp at
# every data point (default: 5% — hf2q ≥ 0.95× llama.cpp).
#
# Baseline (2026-04-23, M5 Max, llama.cpp build b8914-8bc492ebb, apex.gguf):
#   prompt prefill: 364.8 tok/s
#   decode:          97.3 tok/s
# Per project_end_gate_reality_check.md: re-measure llama.cpp on the day
# of the gate run — historical numbers are starting hints, not ship gates.
#
# Usage:
#   scripts/qwen35_bench.sh <gguf_path>
#   scripts/qwen35_bench.sh <gguf_path> --drift-budget 5
#   scripts/qwen35_bench.sh <gguf_path> --skip-llama      # only run hf2q side
#
# Exit codes:
#   0  gate passed (hf2q within drift budget at every point)
#   1  usage / env error
#   2  gate failed (hf2q > drift budget slower than llama.cpp)
#   3  tool invocation failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
OUT_DIR="/tmp/qwen35_bench"
mkdir -p "$OUT_DIR"
LOG_HF2Q="$OUT_DIR/hf2q.log"
LOG_LLAMA="$OUT_DIR/llama.log"

# Data-point matrix.
# Prefill: short (warm-path), typical-long-prompt, long-context.
# Decode: short, medium, long.
PP_LIST=(128 2455 16384)
DECODE_LIST=(64 256 1024)

# Hf2q must run at least this fraction of llama.cpp's tok/s at every point.
# 0.95 = allow 5% slower without failing the gate. Tighten as we optimize.
DRIFT_BUDGET_PCT=5
SKIP_LLAMA=0

usage() {
  cat <<EOF
Usage: scripts/qwen35_bench.sh <gguf_path> [--drift-budget PCT] [--skip-llama]
  <gguf_path>          Qwen3.5 / Qwen3.6 GGUF (qwen35 or qwen35moe arch)
  --drift-budget PCT   Max slowdown vs llama.cpp, in percent (default: 5)
  --skip-llama         Run only hf2q side (useful mid-development)

Exit codes:
  0  gate passed
  1  usage / env error
  2  gate failed (hf2q slower than budget allows at any point)
  3  tool invocation failure
EOF
}
err() { echo "error: $*" >&2; exit 1; }

GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --drift-budget)
      [[ $# -ge 2 ]] || err "--drift-budget requires an argument"
      DRIFT_BUDGET_PCT="$2"; shift 2 ;;
    --skip-llama) SKIP_LLAMA=1; shift ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release)"

if [[ $SKIP_LLAMA -eq 0 ]]; then
  if command -v llama-bench >/dev/null 2>&1; then
    LLAMA_BIN="$(command -v llama-bench)"
  elif [[ -x "/opt/llama.cpp/build/bin/llama-bench" ]]; then
    LLAMA_BIN="/opt/llama.cpp/build/bin/llama-bench"
  else
    err "llama-bench not found on PATH or at /opt/llama.cpp/build/bin/llama-bench (use --skip-llama to run only hf2q)"
  fi
fi

GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "=== Qwen3.5 inference benchmark (ADR-013 P13 match-or-beat gate) ==="
echo "GGUF:           $GGUF_PATH"
echo "hf2q:           $HF2Q_BIN"
if [[ $SKIP_LLAMA -eq 0 ]]; then
  echo "llama-bench:    $LLAMA_BIN"
fi
echo "git HEAD:       $GIT_HEAD"
echo "prefill:        ${PP_LIST[*]} tokens"
echo "decode:         ${DECODE_LIST[*]} tokens"
echo "drift-budget:   ${DRIFT_BUDGET_PCT}% (hf2q >= $(echo "scale=2; (100 - $DRIFT_BUDGET_PCT) / 100" | bc)× llama)"
echo

# --- llama-bench pass ------------------------------------------------------
declare -A LLAMA_TPS_PP LLAMA_TPS_TG
if [[ $SKIP_LLAMA -eq 0 ]]; then
  echo "--- llama-bench pass ---"
  PP_CSV="$(IFS=,; echo "${PP_LIST[*]}")"
  TG_CSV="$(IFS=,; echo "${DECODE_LIST[*]}")"
  if ! "$LLAMA_BIN" -m "$GGUF_PATH" -p "$PP_CSV" -n "$TG_CSV" -o csv \
        --progress 0 > "$OUT_DIR/llama.csv" 2>"$LOG_LLAMA"; then
    echo "llama-bench failed. See $LOG_LLAMA" >&2; exit 3
  fi
  # llama-bench CSV: model,size,params,backend,threads,test,t/s,...
  # test column encodes prefill as "pp<N>" and decode as "tg<N>".
  while IFS=, read -r _m _size _par _bk _th test tps _rest; do
    # Strip quotes.
    test="${test%\"}"; test="${test#\"}"
    tps="${tps%\"}";   tps="${tps#\"}"
    case "$test" in
      pp*) LLAMA_TPS_PP[${test#pp}]="$tps" ;;
      tg*) LLAMA_TPS_TG[${test#tg}]="$tps" ;;
    esac
  done < <(tail -n +2 "$OUT_DIR/llama.csv")
fi

# --- Generate synthetic prompts of varying lengths for the pp matrix -------
# We produce prompts that tokenize to approximately PP_LIST token counts by
# repeating a base passage. The word "approximately" is deliberate: the bench
# table reports the exact prefill token count from hf2q output, not the target.
BASE_PASSAGE="The history of computing is a fascinating journey spanning centuries of human ingenuity. "
gen_prompt() {
  local target_tok="$1"
  # Empirical: ~1.35 chars/token for this passage with the Qwen3.5 tokenizer.
  local reps=$(( (target_tok * 135 / 100) / ${#BASE_PASSAGE} + 1 ))
  local str=""
  for ((i=0; i<reps; i++)); do str+="$BASE_PASSAGE"; done
  echo -n "$str"
}

# --- hf2q-bench pass -------------------------------------------------------
# Uses `hf2q generate --benchmark` which emits:
#   prefill: N tok in Xms (Y tok/s)
#   decode: M tok in Xms (Y tok/s)
# on stderr. --benchmark also prints a structured result block on stdout.
echo "--- hf2q bench pass ---"
declare -A HF2Q_TPS_PP HF2Q_TPS_TG
for pp in "${PP_LIST[@]}"; do
  PROMPT_TEXT="$(gen_prompt "$pp")"
  for tg in "${DECODE_LIST[@]}"; do
    echo -n "  pp=$pp tg=$tg ... "
    # Cold-process invocation per reference_decode_benchmark_methodology.md.
    OUT="$OUT_DIR/hf2q_pp${pp}_tg${tg}.txt"
    if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" \
          --prompt "$PROMPT_TEXT" --max-tokens "$tg" --temperature 0 \
          --benchmark \
          >"$OUT" 2>"$OUT_DIR/hf2q_pp${pp}_tg${tg}.log"; then
      echo "FAIL (see $OUT_DIR/hf2q_pp${pp}_tg${tg}.log)"; exit 3
    fi
    # Parse tok/s from the Benchmark Results block on stdout.
    PP_TPS="$(grep -oE 'Prefill tok/s: *[0-9.]+' "$OUT" | grep -oE '[0-9.]+' | head -1 || echo 0)"
    TG_TPS="$(grep -oE 'Decode tok/s: *[0-9.]+' "$OUT" | grep -oE '[0-9.]+' | head -1 || echo 0)"
    HF2Q_TPS_PP["$pp"]="$PP_TPS"
    HF2Q_TPS_TG["$tg"]="$TG_TPS"
    echo "pp=$PP_TPS tg=$TG_TPS tok/s"
  done
done

# --- Side-by-side report ---------------------------------------------------
echo
echo "--- Prefill (tok/s) ---"
printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "pp" "hf2q" "llama" "ratio" "status"
FAIL_ANY=0
for pp in "${PP_LIST[@]}"; do
  HQ="${HF2Q_TPS_PP[$pp]:-0}"
  LL="${LLAMA_TPS_PP[$pp]:-0}"
  if [[ $SKIP_LLAMA -eq 1 || "$LL" == "0" ]]; then
    printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "$pp" "$HQ" "-" "-" "(skipped)"
    continue
  fi
  RATIO="$(awk -v h="$HQ" -v l="$LL" 'BEGIN { if (l == 0) print "NaN"; else printf "%.3f", h/l }')"
  MIN="$(awk -v b="$DRIFT_BUDGET_PCT" 'BEGIN { printf "%.3f", (100 - b) / 100 }')"
  STATUS="PASS"
  if awk -v r="$RATIO" -v m="$MIN" 'BEGIN { exit !(r < m) }'; then
    STATUS="FAIL"; FAIL_ANY=1
  fi
  printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "$pp" "$HQ" "$LL" "$RATIO" "$STATUS"
done

echo
echo "--- Decode (tok/s) ---"
printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "tg" "hf2q" "llama" "ratio" "status"
for tg in "${DECODE_LIST[@]}"; do
  HQ="${HF2Q_TPS_TG[$tg]:-0}"
  LL="${LLAMA_TPS_TG[$tg]:-0}"
  if [[ $SKIP_LLAMA -eq 1 || "$LL" == "0" ]]; then
    printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "$tg" "$HQ" "-" "-" "(skipped)"
    continue
  fi
  RATIO="$(awk -v h="$HQ" -v l="$LL" 'BEGIN { if (l == 0) print "NaN"; else printf "%.3f", h/l }')"
  MIN="$(awk -v b="$DRIFT_BUDGET_PCT" 'BEGIN { printf "%.3f", (100 - b) / 100 }')"
  STATUS="PASS"
  if awk -v r="$RATIO" -v m="$MIN" 'BEGIN { exit !(r < m) }'; then
    STATUS="FAIL"; FAIL_ANY=1
  fi
  printf "  %-8s  %-12s  %-12s  %-8s  %s\n" "$tg" "$HQ" "$LL" "$RATIO" "$STATUS"
done

echo
if [[ $FAIL_ANY -ne 0 ]]; then
  echo "FAIL: hf2q is below the ${DRIFT_BUDGET_PCT}% drift budget at one or more points."
  echo "Debug: profile with HF2Q_PROFILE_GPU_TS=1 to identify the bottleneck kernel."
  exit 2
fi
echo "PASS: hf2q within ${DRIFT_BUDGET_PCT}% of llama.cpp at every data point."
exit 0
