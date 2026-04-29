#!/usr/bin/env bash
# scripts/iter44-bench-decode-attribution.sh
#
# ADR-015 iter44 — per-CB / per-bucket decode attribution on the
# qwen3.6-35b-a3b-dwq46 −5.96pp coherent-baseline gap.
#
# This is the iter44 cold-SoC capture harness.  Drives:
#   - 5 cold-process trials × NGEN=256 × dwq46 × hf2q with
#     HF2Q_DECODE_PROFILE=1.  Each trial dumps stderr (per-decode-token
#     [GREEDY_PROFILE] lines) for offline aggregation by
#     scripts/iter44-decode-bucket-aggregator.py.
#   - 5 cold-process trials × NGEN=256 × dwq46 × llama-bench (no
#     per-bucket profile available — llama-bench yields whole-decode
#     t/s only).
#   - Per-trial pmset/vm_stat/process-audit gates (matches iter43
#     bench-baseline.sh discipline).
#   - 60s thermal settle between trials (cold-SoC discipline; trial
#     1 is enforce-cold via a leading 120s settle if SETTLE_COLD=1).
#   - mcp-brain-server kill -STOP for the bench window assumed
#     pre-applied by the operator (or call with KILL_BRAIN=1 to do
#     it inline; will CONT on EXIT trap).
#
# The chat-template is the iter41 RAW template (strips chat scaffold
# so the prompt reaches the tokenizer verbatim) — matches the
# coherence_smoke harness; ensures dwq46 produces coherent decode
# rather than the GGUF-default-template Gemma4-fallback gibberish.
#
# Usage:
#   scripts/iter44-bench-decode-attribution.sh [--ngen 256] [--trials 5]
#
# Env:
#   HF2Q_BIN, LLAMA_BENCH_BIN, OUT_DIR, NGEN, N_TRIALS,
#   SETTLE_BETWEEN_SEC, KILL_BRAIN

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/.claude/worktrees/agent-a0557bd3485fb9aa2/target/release/hf2q}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/homebrew/bin/llama-bench}"
MODEL="${MODEL:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf}"
NGEN="${NGEN:-256}"
N_TRIALS="${N_TRIALS:-5}"
PROMPT="${PROMPT:-Hello, my name is}"
# RAW chat template — strip-the-scaffold form so the prompt reaches
# the tokenizer verbatim. Matches tests/coherence_smoke.rs:CHAT_TEMPLATE_RAW.
# NOTE: bash ${VAR:-default} expansion does NOT preserve nested ${...}/{{...}}
# tokens cleanly — define as a plain assignment instead.
if [[ -z "${CHAT_TPL:-}" ]]; then
  CHAT_TPL='{% for message in messages %}{{ message.content }}{% endfor %}'
fi
SETTLE_BETWEEN_SEC="${SETTLE_BETWEEN_SEC:-60}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter44/bench}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
KILL_BRAIN="${KILL_BRAIN:-0}"

mkdir -p "$OUT_DIR"

[[ -x "$HF2Q_BIN" ]] || { echo "FAIL: HF2Q_BIN not executable: $HF2Q_BIN" >&2; exit 1; }
[[ -x "$LLAMA_BENCH_BIN" ]] || { echo "FAIL: LLAMA_BENCH_BIN not executable: $LLAMA_BENCH_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "FAIL: MODEL not found: $MODEL" >&2; exit 1; }

if [[ "$KILL_BRAIN" == "1" ]]; then
  brain_pid=$(pgrep -f mcp-brain-server | head -1 || true)
  if [[ -n "$brain_pid" ]]; then
    echo "[iter44] STOPping mcp-brain-server pid=$brain_pid for bench window"
    kill -STOP "$brain_pid"
    trap 'kill -CONT '"$brain_pid"' 2>/dev/null || true' EXIT INT TERM
  fi
fi

audit() {
  local outpath="$1"
  pmset -g therm > "${outpath}.pmset" 2>&1 || true
  vm_stat > "${outpath}.vm_stat" 2>&1 || true
  ps -ef | grep -E '(hf2q|llama|mlx-native)' | grep -v grep \
    > "${outpath}.ps" 2>&1 || true
  ps -eo user,pid,%cpu,%mem,comm | sort -rn -k3 | head -10 \
    > "${outpath}.ps_top" 2>&1 || true
  pgrep -f mcp-brain-server | xargs -n1 ps -p 2>/dev/null \
    > "${outpath}.brain" 2>&1 || true
}

run_hf2q() {
  local trial="$1"
  local out_base="$OUT_DIR/${DATE_TAG}.hf2q.trial-${trial}"
  echo "  [iter44] hf2q trial $trial -> $out_base"
  audit "${out_base}.pre"
  set +e
  HF2Q_DECODE_PROFILE=1 "$HF2Q_BIN" generate \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$NGEN" \
    --temperature 0 \
    --chat-template "$CHAT_TPL" \
    --benchmark \
      > "${out_base}.stdout" 2> "${out_base}.stderr"
  local rc=$?
  set -e
  audit "${out_base}.post"
  echo "    exit=$rc"
}

run_llama() {
  local trial="$1"
  local out_base="$OUT_DIR/${DATE_TAG}.llama.trial-${trial}"
  echo "  [iter44] llama-bench trial $trial -> $out_base"
  audit "${out_base}.pre"
  set +e
  "$LLAMA_BENCH_BIN" \
    -m "$MODEL" \
    -p 0 \
    -n "$NGEN" \
    -r 3 \
      > "${out_base}.stdout" 2> "${out_base}.stderr"
  local rc=$?
  set -e
  audit "${out_base}.post"
  echo "    exit=$rc"
}

echo "=== ADR-015 iter44 decode-attribution capture ==="
echo "DATE     : $DATE_TAG"
echo "OUT_DIR  : $OUT_DIR"
echo "HF2Q_BIN : $HF2Q_BIN"
echo "LLAMA_BIN: $LLAMA_BENCH_BIN"
echo "MODEL    : $MODEL"
echo "NGEN     : $NGEN"
echo "TRIALS   : $N_TRIALS"

# Wave 1: hf2q trials (cold-SoC, settle between)
echo
echo "--- Wave 1: hf2q ---"
for t in $(seq 1 "$N_TRIALS"); do
  run_hf2q "$t"
  if [[ "$t" -lt "$N_TRIALS" ]]; then
    echo "    settling ${SETTLE_BETWEEN_SEC}s..."
    sleep "$SETTLE_BETWEEN_SEC"
  fi
done

echo
echo "--- inter-wave settle ${SETTLE_BETWEEN_SEC}s ---"
sleep "$SETTLE_BETWEEN_SEC"

# Wave 2: llama trials (cold-SoC, settle between)
echo "--- Wave 2: llama ---"
for t in $(seq 1 "$N_TRIALS"); do
  run_llama "$t"
  if [[ "$t" -lt "$N_TRIALS" ]]; then
    echo "    settling ${SETTLE_BETWEEN_SEC}s..."
    sleep "$SETTLE_BETWEEN_SEC"
  fi
done

# Aggregate hf2q-side per-bucket attribution.
echo
echo "--- aggregation ---"
ls "$OUT_DIR"/${DATE_TAG}.hf2q.trial-*.stderr | sort \
  | xargs python3 /opt/hf2q/.claude/worktrees/agent-a0557bd3485fb9aa2/scripts/iter44-decode-bucket-aggregator.py \
  > "$OUT_DIR/${DATE_TAG}.hf2q.bucket-summary.txt" 2>&1
echo "hf2q bucket summary -> $OUT_DIR/${DATE_TAG}.hf2q.bucket-summary.txt"
cat "$OUT_DIR/${DATE_TAG}.hf2q.bucket-summary.txt"

# Extract llama median t/s from stdouts.
echo
echo "--- llama t/s ---"
for f in "$OUT_DIR"/${DATE_TAG}.llama.trial-*.stdout; do
  awk -v ngen="tg$NGEN" '
    $0 ~ ngen {
      n = split($0, parts, /\| */)
      for (i = 1; i <= n; i++) {
        if (match(parts[i], /[0-9]+\.[0-9]+ *± *[0-9]+\.[0-9]+/)) {
          v = substr(parts[i], RSTART, RLENGTH)
          sub(/ *±.*/, "", v)
          print FILENAME ": " v
          exit
        }
      }
    }
  ' "$f"
done | tee "$OUT_DIR/${DATE_TAG}.llama.tps.txt"

echo
echo "DATE_TAG=$DATE_TAG"
echo "DONE."
