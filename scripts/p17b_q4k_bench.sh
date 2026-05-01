#!/usr/bin/env bash
# ADR-013 P17b — clean cold-process Q4_K bench (replaces P16/P17 single-run noise).
#
# Standing rule (from c5dbc99 retraction):
#   * 3+ cold-process invocations
#   * pre-bench ps/RSS audit; fail if peer ML process active
#   * median + min/max range (not single point)
#   * same-day llama-bench baseline under same conditions
#   * coherence-check output
#
# This script runs hf2q + llama-bench + llama-cli on a Qwen3.5/3.6 GGUF across
# (pp, tg) cells, REPS times each in cold processes, and reports per-cell
# median + range as a markdown table. Apples-to-apples wherever possible:
# hf2q "generate" (prefill+decode in one run) vs llama-cli (prefill+decode in
# one run); llama-bench (synthetic tokens, isolated pp/tg) as cross-check.
#
# Exit codes:
#   0  bench complete (success says nothing about a perf-gate; gate is data-
#      driven and reported in the output table)
#   1  usage / env error
#   2  pre-flight failed (peer ML process or insufficient memory)
#   3  bench tool invocation failed
#
# Receipts land in $OUT_DIR/p17b_summary.md and per-run logs under $OUT_DIR/.

if (( BASH_VERSINFO[0] < 4 )); then
  for candidate in /opt/homebrew/bin/bash /usr/local/bin/bash; do
    if [[ -x "$candidate" ]]; then exec "$candidate" "$0" "$@"; fi
  done
  echo "p17b_q4k_bench.sh requires Bash 4+ ('brew install bash')." >&2; exit 127
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
OUT_DIR="${OUT_DIR:-/tmp/p17b_q4k_bench}"
mkdir -p "$OUT_DIR"

PP_LIST=(31 101 512)
TG_LIST=(64 200)
REPS=3
SKIP_LLAMA=0
SKIP_LLAMA_CLI=0
GGUF_PATH=""

usage() {
  cat <<EOF
Usage: scripts/p17b_q4k_bench.sh <gguf_path> [opts]
  --pp 31,101,512        comma-separated prefill prompt-token lengths
  --tg 64,200            comma-separated decode token counts
  --reps N               cold-process reps per cell (default: 3)
  --skip-llama-bench     skip llama-bench cross-check
  --skip-llama-cli       skip llama-cli apples-to-apples comparison
  --out-dir DIR          override output directory (default: /tmp/p17b_q4k_bench)

Pre-flight:
  Refuses to run if cargo/cc/llama-bench/llama-cli/another hf2q binary or
  any peer ML process (mlx, ollama, lmstudio, vllm, ...) is currently active.

Output:
  \$OUT_DIR/p17b_summary.md   side-by-side median/range table per cell
  \$OUT_DIR/{hf2q,llama-cli,llama-bench}_pp{N}_tg{M}_run{R}.{txt,log}
EOF
}

err() { echo "error: $*" >&2; exit 1; }

# --- Argparse --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --pp)              [[ $# -ge 2 ]] || err "--pp needs arg"; IFS=, read -r -a PP_LIST <<<"$2"; shift 2 ;;
    --tg)              [[ $# -ge 2 ]] || err "--tg needs arg"; IFS=, read -r -a TG_LIST <<<"$2"; shift 2 ;;
    --reps)            [[ $# -ge 2 ]] || err "--reps needs arg"; REPS="$2"; shift 2 ;;
    --skip-llama-bench) SKIP_LLAMA=1; shift ;;
    --skip-llama-cli)   SKIP_LLAMA_CLI=1; shift ;;
    --out-dir)         [[ $# -ge 2 ]] || err "--out-dir needs arg"; OUT_DIR="$2"; mkdir -p "$OUT_DIR"; shift 2 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional: $1"; GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release --bin hf2q)"

LLAMA_BENCH_BIN=""
LLAMA_COMPLETION_BIN=""
if [[ $SKIP_LLAMA -eq 0 ]]; then
  if   [[ -x "/opt/llama.cpp/build/bin/llama-bench" ]]; then LLAMA_BENCH_BIN="/opt/llama.cpp/build/bin/llama-bench"
  elif command -v llama-bench >/dev/null 2>&1;            then LLAMA_BENCH_BIN="$(command -v llama-bench)"
  else err "llama-bench not found (use --skip-llama-bench to skip)"
  fi
fi
if [[ $SKIP_LLAMA_CLI -eq 0 ]]; then
  # llama-cli is interactive-only on recent homebrew (b8807+); use llama-completion
  # for non-interactive single-shot prefill+decode runs.
  if   [[ -x "/opt/llama.cpp/build/bin/llama-completion" ]]; then LLAMA_COMPLETION_BIN="/opt/llama.cpp/build/bin/llama-completion"
  elif command -v llama-completion >/dev/null 2>&1;             then LLAMA_COMPLETION_BIN="$(command -v llama-completion)"
  elif [[ -x "/opt/llama.cpp/build/bin/llama-cli" ]];           then LLAMA_COMPLETION_BIN="/opt/llama.cpp/build/bin/llama-cli"
  elif command -v llama-cli >/dev/null 2>&1;                    then LLAMA_COMPLETION_BIN="$(command -v llama-cli)"
  else err "neither llama-completion nor llama-cli found (use --skip-llama-cli to skip)"
  fi
fi

# --- Pre-flight: ps audit + vm_stat ---------------------------------------
echo "=== P17b pre-flight ==="
SELF_PID=$$

# Build a list of competing processes — anything running cargo/cc-build, or any
# OTHER hf2q/llama-bench/llama-cli/mlx/ollama/lmstudio/vllm process. The script
# is tolerant of /Users/.*/.local/bin/claude (the harness spawner) but not of
# concurrent ML compute.
COMPETING="$(ps -axwwo pid=,command= \
  | awk -v self=$SELF_PID '
      $1 != self && $1 != self+1 {
        line = $0
        sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "", line)
        # Match anything that looks like a build / inference process.
        if (line ~ /(^|\/)cargo([[:space:]]|$)/   ||
            line ~ /\/rustc([[:space:]]|$)/      ||
            line ~ /(^|\/)cc1?([[:space:]]|$)/   ||
            line ~ /(^|\/)clang(\+\+)?([[:space:]]|$)/ ||
            line ~ /(^|\/)hf2q([[:space:]]|$)/   ||
            line ~ /(^|\/)llama-(bench|cli|server)([[:space:]]|$)/ ||
            line ~ /(^|\/)mlx_lm/                ||
            line ~ /(^|\/)ollama([[:space:]]|$)/ ||
            line ~ /(^|\/)lmstudio/              ||
            line ~ /(^|\/)vllm/) {
          print line
        }
      }' || true)"

if [[ -n "$COMPETING" ]]; then
  echo "BLOCKED: competing ML/build processes detected:" >&2
  echo "$COMPETING" | sed 's/^/  /' >&2
  echo "(per feedback_bench_process_audit — refuse to bench under contention)" >&2
  exit 2
fi
echo "  ps audit: clean (no competing ML/build processes)"

PAGE_BYTES=$(vm_stat | awk -F'of ' '/page size of/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
PAGES_FREE=$(vm_stat | awk -F: '/Pages free/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
PAGES_INACTIVE=$(vm_stat | awk -F: '/Pages inactive/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
FREE_GB=$(awk -v p="$PAGES_FREE" -v i="$PAGES_INACTIVE" -v sz="$PAGE_BYTES" 'BEGIN { printf "%.1f", (p+i)*sz/1073741824 }')
echo "  vm_stat: ${FREE_GB} GB free+inactive"
if (( $(awk -v g="$FREE_GB" 'BEGIN { print (g < 30) }') )); then
  echo "WARNING: < 30 GB free; bench results may be contaminated by paging." >&2
fi

GIT_HEAD_HF2Q="$(git -C /opt/hf2q rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_HEAD_MLX="$(git -C /opt/mlx-native rev-parse HEAD 2>/dev/null || echo unknown)"
GGUF_BYTES="$(stat -f '%z' "$GGUF_PATH" 2>/dev/null || stat -c '%s' "$GGUF_PATH" 2>/dev/null || echo 0)"
GGUF_GB="$(awk -v b="$GGUF_BYTES" 'BEGIN { printf "%.1f", b/1073741824 }')"

echo
echo "GGUF:           $GGUF_PATH (${GGUF_GB} GB)"
echo "hf2q:           $GIT_HEAD_HF2Q"
echo "mlx-native:     $GIT_HEAD_MLX"
echo "reps:           $REPS"
echo "pp matrix:      ${PP_LIST[*]}"
echo "tg matrix:      ${TG_LIST[*]}"
echo "out:            $OUT_DIR"
echo

# --- Helpers --------------------------------------------------------------
# Synthesize a prompt of approximately N tokens. Empirical 1.35 chars/tok for
# Qwen3.5 BPE on plain English. Actual prefill_n is parsed from hf2q stderr.
BASE_PASSAGE="The history of computing is a fascinating journey spanning centuries of human ingenuity. "
gen_prompt() {
  local target="$1"
  local reps=$(( (target * 135 / 100) / ${#BASE_PASSAGE} + 1 ))
  local str=""
  local i
  for ((i=0; i<reps; i++)); do str+="$BASE_PASSAGE"; done
  echo -n "$str"
}

# Compute median, min, max of stdin (one number per line). Empty input prints "-,-,-".
# macOS-portable: uses sort + awk (no asort()).
stats3() {
  local sorted
  sorted="$(grep -E '^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$' | LC_ALL=C sort -g)"
  if [[ -z "$sorted" ]]; then echo "-,-,-"; return; fi
  awk 'BEGIN { n = 0 }
       { v[++n] = $1 + 0 }
       END {
         m = (n % 2 == 1) ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2
         printf "%.2f,%.2f,%.2f\n", m, v[1], v[n]
       }' <<<"$sorted"
}

# --- hf2q runs ------------------------------------------------------------
declare -A HF2Q_PP_RUNS HF2Q_TG_RUNS
echo "--- hf2q (cold-process, $REPS reps per cell) ---"
for pp in "${PP_LIST[@]}"; do
  for tg in "${TG_LIST[@]}"; do
    PROMPT="$(gen_prompt "$pp")"
    PP_VALS="" TG_VALS=""
    for ((r=1; r<=REPS; r++)); do
      OUT="$OUT_DIR/hf2q_pp${pp}_tg${tg}_run${r}.txt"
      LOG="$OUT_DIR/hf2q_pp${pp}_tg${tg}_run${r}.log"
      if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" \
            --prompt "$PROMPT" --max-tokens "$tg" --temperature 0 \
            --benchmark > "$OUT" 2> "$LOG"; then
        echo "  hf2q FAIL pp=$pp tg=$tg run=$r — see $LOG" >&2
        exit 3
      fi
      PP_TPS="$(grep -oE 'Prefill tok/s: *[0-9.]+' "$OUT" | head -1 | grep -oE '[0-9.]+' | head -1)"
      TG_TPS="$(grep -oE 'Decode tok/s: *[0-9.]+'  "$OUT" | head -1 | grep -oE '[0-9.]+' | head -1)"
      [[ -n "${PP_TPS:-}" ]] && PP_VALS+="$PP_TPS"$'\n'
      [[ -n "${TG_TPS:-}" ]] && TG_VALS+="$TG_TPS"$'\n'
      printf "  pp=%-4s tg=%-4s run=%d  pp_tps=%s  tg_tps=%s\n" "$pp" "$tg" "$r" "${PP_TPS:-MISS}" "${TG_TPS:-MISS}"
    done
    HF2Q_PP_RUNS["$pp/$tg"]="$(printf '%s' "$PP_VALS" | stats3)"
    HF2Q_TG_RUNS["$pp/$tg"]="$(printf '%s' "$TG_VALS" | stats3)"
  done
done

# --- llama-completion (apples-to-apples: prefill+decode in one run) -------
declare -A LLAMA_CLI_PP_RUNS LLAMA_CLI_TG_RUNS
if [[ $SKIP_LLAMA_CLI -eq 0 ]]; then
  echo
  echo "--- $(basename "$LLAMA_COMPLETION_BIN") (cold-process, $REPS reps per cell) ---"
  for pp in "${PP_LIST[@]}"; do
    for tg in "${TG_LIST[@]}"; do
      PROMPT="$(gen_prompt "$pp")"
      PP_VALS="" TG_VALS=""
      for ((r=1; r<=REPS; r++)); do
        OUT="$OUT_DIR/llama-completion_pp${pp}_tg${tg}_run${r}.txt"
        LOG="$OUT_DIR/llama-completion_pp${pp}_tg${tg}_run${r}.log"
        # llama-completion: non-interactive single-shot equivalent of legacy
        # llama-cli -no-cnv. -ngl 99 = full-GPU offload; --simple-io disables
        # color/tty escapes; -no-cnv kept for older binaries that still take it.
        # Some builds emit perf lines to stdout, others to stderr; we grep both.
        if ! "$LLAMA_COMPLETION_BIN" -m "$GGUF_PATH" --prompt "$PROMPT" -n "$tg" \
              --temp 0 --seed 42 -ngl 99 --no-warmup --simple-io \
              > "$OUT" 2> "$LOG" </dev/null; then
          echo "  llama-completion FAIL pp=$pp tg=$tg run=$r — see $LOG" >&2
          exit 3
        fi
        # llama_perf_context_print: prompt eval time =     X ms / N tokens (X.XX ms per token, Y tokens per second)
        # llama_perf_context_print:        eval time =     X ms / M runs   (X.XX ms per token, Y tokens per second)
        PP_TPS="$(grep -h 'prompt eval time' "$LOG" "$OUT" 2>/dev/null | head -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)"
        # Decode line: "<prefix>:        eval time = ..." with no "prompt".
        # Match by the unique " eval time =" substring while excluding "prompt eval".
        TG_TPS="$(grep -h ' eval time =' "$LOG" "$OUT" 2>/dev/null | grep -v 'prompt eval' | head -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | head -1)"
        [[ -n "${PP_TPS:-}" ]] && PP_VALS+="$PP_TPS"$'\n'
        [[ -n "${TG_TPS:-}" ]] && TG_VALS+="$TG_TPS"$'\n'
        printf "  pp=%-4s tg=%-4s run=%d  pp_tps=%s  tg_tps=%s\n" "$pp" "$tg" "$r" "${PP_TPS:-MISS}" "${TG_TPS:-MISS}"
      done
      LLAMA_CLI_PP_RUNS["$pp/$tg"]="$(printf '%s' "$PP_VALS" | stats3)"
      LLAMA_CLI_TG_RUNS["$pp/$tg"]="$(printf '%s' "$TG_VALS" | stats3)"
    done
  done
fi

# --- llama-bench (canonical reference: synthetic tokens, isolated pp/tg) --
declare -A LLAMA_BENCH_PP_RUNS LLAMA_BENCH_TG_RUNS
if [[ $SKIP_LLAMA -eq 0 ]]; then
  echo
  echo "--- llama-bench (cold-process, REPS=$REPS, internal reps=1 each) ---"
  PP_CSV="$(IFS=,; echo "${PP_LIST[*]}")"
  TG_CSV="$(IFS=,; echo "${TG_LIST[*]}")"
  declare -A PP_RUNS TG_RUNS
  for ((r=1; r<=REPS; r++)); do
    OUT="$OUT_DIR/llama-bench_run${r}.csv"
    LOG="$OUT_DIR/llama-bench_run${r}.log"
    if ! "$LLAMA_BENCH_BIN" -m "$GGUF_PATH" -p "$PP_CSV" -n "$TG_CSV" \
          -r 1 -o csv > "$OUT" 2> "$LOG"; then
      echo "  llama-bench FAIL run=$r — see $LOG" >&2
      exit 3
    fi
    # Newer llama-bench CSV (b8680+) has ~40 columns; locate by header name.
    while IFS=, read -r kind n tps; do
      case "$kind" in
        pp) PP_RUNS["$n"]+="$tps"$'\n' ;;
        tg) TG_RUNS["$n"]+="$tps"$'\n' ;;
      esac
    done < <(python3 -c "
import csv, sys
with open('$OUT') as fp:
    rdr = csv.DictReader(fp)
    for r in rdr:
        np_, ng = int(r.get('n_prompt') or 0), int(r.get('n_gen') or 0)
        ts = r.get('avg_ts') or '0'
        if np_ > 0:
            print(f'pp,{np_},{ts}')
        elif ng > 0:
            print(f'tg,{ng},{ts}')
")
    echo "  llama-bench run $r: $(wc -l <"$OUT" | tr -d ' ') CSV rows"
  done
  for pp in "${PP_LIST[@]}"; do
    LLAMA_BENCH_PP_RUNS["$pp"]="$(printf '%s' "${PP_RUNS[$pp]:-}" | stats3)"
  done
  for tg in "${TG_LIST[@]}"; do
    LLAMA_BENCH_TG_RUNS["$tg"]="$(printf '%s' "${TG_RUNS[$tg]:-}" | stats3)"
  done
fi

# --- Coherence check ------------------------------------------------------
echo
echo "--- Coherence check ---"
COH_PROMPT="Explain what a transformer neural network is in 3 sentences."
COH_OUT="$OUT_DIR/coherence_hf2q.txt"
COH_LOG="$OUT_DIR/coherence_hf2q.log"
"$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$COH_PROMPT" \
  --max-tokens 80 --temperature 0 > "$COH_OUT" 2> "$COH_LOG" \
  || { echo "  hf2q coherence run FAILED — see $COH_LOG" >&2; exit 3; }
# Strip ANSI and accept printable ASCII + common Unicode (UTF-8 byte-aware).
ASCII_RATIO="$(LC_ALL=C tr -dc '[:print:]\n' < "$COH_OUT" | wc -c | awk -v t="$(wc -c <"$COH_OUT")" '{ if (t==0) print 0; else printf "%.3f", $1/t }')"
HEAD120="$(head -c 200 "$COH_OUT" | tr -d '\n' | head -c 120)"
echo "  prompt:        $COH_PROMPT"
echo "  head(120):     $HEAD120"
echo "  ascii_ratio:   $ASCII_RATIO  (expect >= 0.85 for coherent English)"

# --- Render markdown summary ---------------------------------------------
SUMMARY="$OUT_DIR/p17b_summary.md"
{
  echo "# ADR-013 P17b — clean Q4_K bench"
  echo
  echo "- date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "- gguf: \`$GGUF_PATH\` (${GGUF_GB} GB)"
  echo "- hf2q HEAD: \`$GIT_HEAD_HF2Q\`"
  echo "- mlx-native HEAD: \`$GIT_HEAD_MLX\`"
  echo "- reps per cell: $REPS"
  echo "- coherence ascii_ratio: $ASCII_RATIO"
  echo
  echo "## Prefill (tok/s) — median \`(min .. max)\`"
  echo
  printf '| pp | tg | hf2q | llama-cli | llama-bench | hf2q/llama-cli | hf2q/llama-bench |\n'
  printf '|---|---|---|---|---|---|---|\n'
  for pp in "${PP_LIST[@]}"; do
    for tg in "${TG_LIST[@]}"; do
      IFS=, read -r HM HL HH <<<"${HF2Q_PP_RUNS["$pp/$tg"]:-,,}"
      IFS=, read -r CM CL CH <<<"${LLAMA_CLI_PP_RUNS["$pp/$tg"]:-,,}"
      IFS=, read -r BM BL BH <<<"${LLAMA_BENCH_PP_RUNS["$pp"]:-,,}"
      RC="-"; [[ "$HM" != "-" && "$CM" != "-" && "$CM" != "0.00" ]] && RC=$(awk -v h="$HM" -v l="$CM" 'BEGIN { if (l+0 == 0) { print "-" } else { printf "%.2fx", h/l } }')
      RB="-"; [[ "$HM" != "-" && "$BM" != "-" && "$BM" != "0.00" ]] && RB=$(awk -v h="$HM" -v l="$BM" 'BEGIN { if (l+0 == 0) { print "-" } else { printf "%.2fx", h/l } }')
      printf '| %s | %s | %s (%s..%s) | %s (%s..%s) | %s (%s..%s) | %s | %s |\n' \
        "$pp" "$tg" "$HM" "$HL" "$HH" "$CM" "$CL" "$CH" "$BM" "$BL" "$BH" "$RC" "$RB"
    done
  done
  echo
  echo "## Decode (tok/s) — median \`(min .. max)\`"
  echo
  printf '| pp | tg | hf2q | llama-cli | llama-bench | hf2q/llama-cli | hf2q/llama-bench |\n'
  printf '|---|---|---|---|---|---|---|\n'
  for pp in "${PP_LIST[@]}"; do
    for tg in "${TG_LIST[@]}"; do
      IFS=, read -r HM HL HH <<<"${HF2Q_TG_RUNS["$pp/$tg"]:-,,}"
      IFS=, read -r CM CL CH <<<"${LLAMA_CLI_TG_RUNS["$pp/$tg"]:-,,}"
      IFS=, read -r BM BL BH <<<"${LLAMA_BENCH_TG_RUNS["$tg"]:-,,}"
      RC="-"; [[ "$HM" != "-" && "$CM" != "-" && "$CM" != "0.00" ]] && RC=$(awk -v h="$HM" -v l="$CM" 'BEGIN { if (l+0 == 0) { print "-" } else { printf "%.2fx", h/l } }')
      RB="-"; [[ "$HM" != "-" && "$BM" != "-" && "$BM" != "0.00" ]] && RB=$(awk -v h="$HM" -v l="$BM" 'BEGIN { if (l+0 == 0) { print "-" } else { printf "%.2fx", h/l } }')
      printf '| %s | %s | %s (%s..%s) | %s (%s..%s) | %s (%s..%s) | %s | %s |\n' \
        "$pp" "$tg" "$HM" "$HL" "$HH" "$CM" "$CL" "$CH" "$BM" "$BL" "$BH" "$RC" "$RB"
    done
  done
} > "$SUMMARY"

cat "$SUMMARY"
echo
echo "Receipts: $SUMMARY"
exit 0
