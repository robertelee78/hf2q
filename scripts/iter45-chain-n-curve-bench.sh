#!/usr/bin/env bash
# scripts/iter45-chain-n-curve-bench.sh
#
# ADR-015 iter45 — chain_n N-curve recapture on coherent baseline.
#
# Sweeps HF2Q_PARTIAL_CHAIN_N over {1, 2, 4, 8, 20} for each of the 4
# coherent-baseline fixtures (dwq46, apex, 27b-dwq46, gemma-26B) and
# captures cold-SoC paired hf2q + llama-bench measurements at NGEN=256.
#
# Per-quant-class default lookup table (`forward_gpu.rs:315-329` at HEAD
# c4f63f9):
#   - DenseQ + Q4_K (no MoE):  cn = 4   (27B-dwq46)
#   - MoeQ   + Q4_K:           cn = 2   (35B-dwq46)
#   - MoeQ   + Q5_K / Q6_K:    cn = 1   (35B-apex)
#   - everything else:         cn = 1   (gemma-26B uses forward_mlx;
#                                       lever does not apply — included
#                                       here as control to prove no-effect)
#
# THIS HARNESS DRIVES THE BENCH; IT DOES NOT RECOMMEND CODE CHANGES.
# The Phase 5 gate in iter45 (ship only if a winner beats current
# default by ≥1pp on its primary fixture AND ≥0pp on every other
# fixture AND coherence_smoke 12/12 PASS) is enforced by the operator
# reviewing the aggregated tables.
#
# Methodology — matches iter44 exactly:
#   - 5 cold-process trials × NGEN=256 × hf2q with HF2Q_DECODE_PROFILE=1
#     and HF2Q_PARTIAL_CHAIN_N=$N forced (overrides forward_gpu.rs:2131
#     env-read).
#   - 5 cold-process trials × NGEN=256 × llama-bench (no env; n_cb=1
#     default per ggml-metal-context.m:19).  Llama is benched ONCE per
#     fixture (not per N) because the baseline is independent of hf2q's
#     chain_n setting.
#   - 60s thermal settle between trials; 120s between fixtures.
#   - Per-trial pmset/vm_stat/process-audit gates; archived under
#     /tmp/adr015-iter45/bench/<fixture>/cn-<N>/.
#   - mcp-brain-server kill -STOP for the bench window (KILL_BRAIN=1).
#   - PRE-FLIGHT FCP / DAVINCI / FFMPEG / HANDBRAKE GATE — the iter44
#     entry surfaced Final Cut Pro at 195% sustained CPU contaminating
#     hf2q trials.  This harness ABORTS if any of those processes is
#     detected; the operator must close them before bench.  Override
#     with SKIP_FCP_GATE=1 (NOT recommended; produces noisy data).
#
# Time budget: 4 fixtures × 5 N values × (5 hf2q trials + 60s settle)
#              + 4 fixtures × 5 llama trials + 60s settle.  Approx
#              4 × (5 × 5 × 90s + 5 × 60s + ~3 min compute/N) = ~3.5h
#              wall clock with cold-SoC discipline.
#
# Output:
#   /tmp/adr015-iter45/bench/<fixture>/cn-<N>/<DATE>.{hf2q,llama}.trial-N.{stdout,stderr,pre.*,post.*}
#   /tmp/adr015-iter45/bench/<fixture>/cn-<N>/<DATE>.hf2q.bucket-summary.txt
#   /tmp/adr015-iter45/bench/<fixture>/cn-<N>/<DATE>.llama.tps.txt
#   /tmp/adr015-iter45/N-curve-summary.tsv  (final aggregated table)
#
# Usage:
#   scripts/iter45-chain-n-curve-bench.sh [--fixtures dwq46,...] [--ns 1,2,4,8,20]
#
# Env:
#   HF2Q_BIN, LLAMA_BENCH_BIN, OUT_DIR, NGEN, N_TRIALS, SETTLE_BETWEEN_SEC,
#   KILL_BRAIN, SKIP_FCP_GATE
#
# Exit codes:
#   0 — bench completed all cells
#   1 — environment / binary / model missing
#   2 — FCP/DaVinci/Handbrake/ffmpeg detected and SKIP_FCP_GATE != 1
#   3 — thermal flag detected mid-bench
#   4 — RAM headroom insufficient

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/.claude/worktrees/agent-aa7977d8cd65c5cef/target/release/hf2q}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/homebrew/bin/llama-bench}"
NGEN="${NGEN:-256}"
N_TRIALS="${N_TRIALS:-5}"
PROMPT="${PROMPT:-Hello, my name is}"
SETTLE_BETWEEN_SEC="${SETTLE_BETWEEN_SEC:-60}"
SETTLE_FIXTURE_SEC="${SETTLE_FIXTURE_SEC:-120}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter45/bench}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
KILL_BRAIN="${KILL_BRAIN:-0}"
SKIP_FCP_GATE="${SKIP_FCP_GATE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

# RAW chat template — matches tests/coherence_smoke.rs:CHAT_TEMPLATE_RAW
CHAT_TPL='{% for message in messages %}{{ message.content }}{% endfor %}'

# Default sweep set; can be overridden via --ns
NS_DEFAULT="1,2,4,8,20"
FIXTURES_DEFAULT="dwq46,apex,27b-dwq46,gemma-26B"

NS="$NS_DEFAULT"
FIXTURES="$FIXTURES_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns) NS="$2"; shift 2 ;;
    --fixtures) FIXTURES="$2"; shift 2 ;;
    -h|--help) sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ----- fixture map: name → gguf path -----
fixture_path() {
  case "$1" in
    dwq46)
      echo "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf"
      ;;
    apex)
      echo "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf"
      ;;
    27b-dwq46)
      echo "/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf"
      ;;
    gemma-26B)
      echo "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
      ;;
    *) echo "unknown fixture: $1" >&2; return 1 ;;
  esac
}

# ----- pre-flight FCP / heavyweight gate (FIRST: contamination is fatal) -----
if [[ "$SKIP_FCP_GATE" != "1" ]]; then
  contaminator="$(ps -ef | egrep -i '(final cut|finalcut|fcp|davinci|handbrake|ffmpeg)' | grep -v grep || true)"
  if [[ -n "$contaminator" ]]; then
    echo "FAIL: heavyweight CPU contaminator detected — abort per iter44 methodology fix:" >&2
    echo "$contaminator" >&2
    echo "  (set SKIP_FCP_GATE=1 to override; produces noisy hf2q data per iter44 trial-1/5)" >&2
    exit 2
  fi
fi

[[ -x "$HF2Q_BIN" ]] || { echo "FAIL: HF2Q_BIN not executable: $HF2Q_BIN" >&2; exit 1; }
[[ -x "$LLAMA_BENCH_BIN" ]] || { echo "FAIL: LLAMA_BENCH_BIN not executable" >&2; exit 1; }

# ----- pre-flight thermal + RAM gate -----
if pmset -g therm 2>&1 | grep -q "CPU_Speed_Limit"; then
  limit=$(pmset -g therm 2>&1 | awk -F'=' '/CPU_Speed_Limit/ {print $2}' | tr -d ' ')
  if [[ -n "$limit" && "$limit" != "100" ]]; then
    echo "FAIL: thermal throttle detected (CPU_Speed_Limit=$limit)" >&2
    exit 3
  fi
fi
PAGE_SIZE=$(vm_stat | head -1 | awk -F'of ' '{print $2}' | tr -d ' bytes)')
FREE=$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')
INACT=$(vm_stat | awk '/Pages inactive/ {print $3}' | tr -d '.')
SPEC=$(vm_stat | awk '/Pages speculative/ {print $3}' | tr -d '.')
TOTAL_GB=$(( ($FREE + $INACT + $SPEC) * $PAGE_SIZE / 1024 / 1024 / 1024 ))
echo "RAM headroom: ${TOTAL_GB} GB (need >= ${MIN_FREE_GB})"
if [[ $TOTAL_GB -lt $MIN_FREE_GB ]]; then
  echo "FAIL: insufficient RAM" >&2; exit 4
fi

# ----- mcp-brain STOP/CONT -----
if [[ "$KILL_BRAIN" == "1" ]]; then
  brain_pid=$(pgrep -f mcp-brain-server | head -1 || true)
  if [[ -n "$brain_pid" ]]; then
    echo "[iter45] STOPping mcp-brain-server pid=$brain_pid"
    kill -STOP "$brain_pid"
    trap 'kill -CONT '"$brain_pid"' 2>/dev/null || true' EXIT INT TERM
  fi
fi

mkdir -p "$OUT_DIR"

audit() {
  local outpath="$1"
  pmset -g therm > "${outpath}.pmset" 2>&1 || true
  vm_stat > "${outpath}.vm_stat" 2>&1 || true
  ps -eo user,pid,%cpu,%mem,comm | sort -rn -k3 | head -10 \
    > "${outpath}.ps_top" 2>&1 || true
  ps -ef | egrep -i '(final cut|finalcut|fcp|davinci|handbrake|ffmpeg|hf2q|llama|mlx-native|mcp-brain)' \
    | grep -v grep > "${outpath}.ps" 2>&1 || true
}

run_hf2q_cell() {
  local fixture="$1"
  local cn="$2"
  local trial="$3"
  local cell_dir="$4"
  local model="$5"
  local out_base="$cell_dir/${DATE_TAG}.hf2q.trial-${trial}"
  echo "    [hf2q cn=$cn trial $trial] -> $out_base"
  audit "${out_base}.pre"
  set +e
  HF2Q_DECODE_PROFILE=1 \
  HF2Q_PARTIAL_CHAIN_N="$cn" \
  "$HF2Q_BIN" generate \
    --model "$model" \
    --prompt "$PROMPT" \
    --max-tokens "$NGEN" \
    --temperature 0 \
    --chat-template "$CHAT_TPL" \
    --benchmark \
      > "${out_base}.stdout" 2> "${out_base}.stderr"
  local rc=$?
  set -e
  audit "${out_base}.post"
  echo "      exit=$rc"
}

run_llama_cell() {
  local fixture="$1"
  local trial="$2"
  local cell_dir="$3"
  local model="$4"
  local out_base="$cell_dir/${DATE_TAG}.llama.trial-${trial}"
  echo "    [llama trial $trial] -> $out_base"
  audit "${out_base}.pre"
  set +e
  "$LLAMA_BENCH_BIN" \
    -m "$model" \
    -p 0 \
    -n "$NGEN" \
    -r 3 \
      > "${out_base}.stdout" 2> "${out_base}.stderr"
  local rc=$?
  set -e
  audit "${out_base}.post"
  echo "      exit=$rc"
}

aggregate_cell() {
  local cell_dir="$1"
  local fixture="$2"
  local cn="$3"
  # hf2q bucket summary (uses iter44 aggregator)
  ls "$cell_dir"/${DATE_TAG}.hf2q.trial-*.stderr 2>/dev/null \
    | sort \
    | xargs python3 /opt/hf2q/.claude/worktrees/agent-aa7977d8cd65c5cef/scripts/iter44-decode-bucket-aggregator.py \
      > "$cell_dir/${DATE_TAG}.hf2q.bucket-summary.txt" 2>&1 || true
  # llama tps
  : > "$cell_dir/${DATE_TAG}.llama.tps.txt"
  for f in "$cell_dir"/${DATE_TAG}.llama.trial-*.stdout; do
    [[ -f "$f" ]] || continue
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
    ' "$f" >> "$cell_dir/${DATE_TAG}.llama.tps.txt"
  done
}

echo "=== ADR-015 iter45 chain_n N-curve recapture ==="
echo "DATE     : $DATE_TAG"
echo "OUT_DIR  : $OUT_DIR"
echo "HF2Q_BIN : $HF2Q_BIN"
echo "FIXTURES : $FIXTURES"
echo "NS       : $NS"
echo "TRIALS   : $N_TRIALS"

IFS=',' read -ra FIXTURE_ARR <<< "$FIXTURES"
IFS=',' read -ra N_ARR <<< "$NS"

for fixture in "${FIXTURE_ARR[@]}"; do
  model="$(fixture_path "$fixture")"
  if [[ ! -f "$model" ]]; then
    echo "SKIP fixture $fixture — model not found at $model" >&2
    continue
  fi
  fixture_dir="$OUT_DIR/$fixture"
  mkdir -p "$fixture_dir"

  # Run llama once per fixture (independent of cn).
  llama_dir="$fixture_dir/llama-base"
  mkdir -p "$llama_dir"
  echo
  echo "--- fixture $fixture / llama baseline ---"
  for t in $(seq 1 "$N_TRIALS"); do
    run_llama_cell "$fixture" "$t" "$llama_dir" "$model"
    [[ $t -lt $N_TRIALS ]] && sleep "$SETTLE_BETWEEN_SEC"
  done
  aggregate_cell "$llama_dir" "$fixture" "llama"

  for cn in "${N_ARR[@]}"; do
    cell_dir="$fixture_dir/cn-$cn"
    mkdir -p "$cell_dir"
    echo
    echo "--- fixture $fixture / cn=$cn ---"
    sleep "$SETTLE_FIXTURE_SEC"
    for t in $(seq 1 "$N_TRIALS"); do
      run_hf2q_cell "$fixture" "$cn" "$t" "$cell_dir" "$model"
      [[ $t -lt $N_TRIALS ]] && sleep "$SETTLE_BETWEEN_SEC"
    done
    aggregate_cell "$cell_dir" "$fixture" "$cn"
  done
done

# ----- final N-curve summary -----
SUMMARY="$OUT_DIR/N-curve-summary-${DATE_TAG}.tsv"
echo "fixture	cn	hf2q_med	llama_med	ratio	n_cb_observed" > "$SUMMARY"
for fixture in "${FIXTURE_ARR[@]}"; do
  fixture_dir="$OUT_DIR/$fixture"
  llama_med="$(awk -F': ' '{print $2}' "$fixture_dir/llama-base/${DATE_TAG}.llama.tps.txt" 2>/dev/null \
                | sort -n | awk 'NR==int(NR/2)+1 {print; exit}' || echo NaN)"
  for cn in "${N_ARR[@]}"; do
    cell_dir="$fixture_dir/cn-$cn"
    bs="$cell_dir/${DATE_TAG}.hf2q.bucket-summary.txt"
    [[ -f "$bs" ]] || continue
    hf2q_med="$(grep -E '^  tps' "$bs" | head -1 | awk '{print $3}' | sed 's/median=//' || echo NaN)"
    n_cb="$(awk '/^  cb / {print $3}' "$bs" 2>/dev/null | head -1 || echo NaN)"
    ratio="NaN"
    if [[ "$hf2q_med" != "NaN" && "$llama_med" != "NaN" ]]; then
      ratio="$(python3 -c "print(f'{float('$hf2q_med')/float('$llama_med'):.4f}')")"
    fi
    echo "$fixture	$cn	$hf2q_med	$llama_med	$ratio	$n_cb" >> "$SUMMARY"
  done
done

echo
echo "=== iter45 N-curve summary -> $SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo
echo "DATE_TAG=$DATE_TAG"
echo "DONE."
