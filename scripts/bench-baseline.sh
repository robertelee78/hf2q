#!/usr/bin/env bash
# scripts/bench-baseline.sh
#
# ADR-015 cold-SoC baseline benchmark harness.  Used for:
#   - P0   gemma baseline (iter4)
#   - P6   per-family bench gate (post-P3 single-CB rewrite)
#   - Wave 2b hard gate #1 — apex 35B-A3B MoE re-measurement
#
# Methodology (per `feedback_perf_gate_thermal_methodology` +
# `feedback_check_ram_before_inference` + ADR-015 D4):
#
#   - Cold SoC.  `pmset -g therm` checked pre-trial; if
#     CPU_Speed_Limit ≠ 100 the trial aborts with status 2.
#   - 60-second thermal settle BETWEEN trials (`THERMAL_SETTLE_SEC`).
#   - RAM-headroom precheck.  vm_stat free+inactive+speculative pages
#     must exceed `MIN_FREE_GB * 1024^3 / page_size` before the harness
#     loads any model.  Default MIN_FREE_GB=30 (apex 35B-A3B working
#     set; gemma-26B + qwen-27B are smaller, so this default is safe).
#   - 3 cold-process trials per side by default
#     (`hf2q --benchmark`'s own internal 5 runs are a separate axis;
#     see `reference_decode_benchmark_methodology`).  N_TRIALS=5 is
#     the Wave 2b methodology for outlier-robust median.
#   - n_gen = 256 (D4 reference decode length).
#   - greedy decode (`--temperature 0`).
#   - Same prompt for both sides ("Hello, my name is" — matches
#     §P3a' fixture).  llama-bench uses `-p 0 -n 256` with no prompt
#     warmup; hf2q --benchmark uses the same prompt + n_gen.
#
# Outputs:
#   ${OUT_DIR}/baseline-${LABEL}-${DATE}.metadata.json
#   ${OUT_DIR}/baseline-${LABEL}-${DATE}.hf2q.{trial-1,2,...}.{stdout,stderr}
#   ${OUT_DIR}/baseline-${LABEL}-${DATE}.llama.{trial-1,2,...}.{stdout,stderr}
#   ${OUT_DIR}/baseline-${LABEL}-${DATE}.summary.txt   (median per side
#                                                       + ratio)
#
# Usage:
#   scripts/bench-baseline.sh \
#     --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
#     --label gemma-26B-dwq-p0
#
# Optional env:
#   N_TRIALS=5 NGEN=256 PROMPT="..." MIN_FREE_GB=30 \
#   THERMAL_SETTLE_SEC=60 SKIP_THERMAL_GATE=1 SKIP_RAM_GATE=1 \
#   HF2Q_BIN=/path/to/hf2q LLAMA_BENCH_BIN=/path/to/llama-bench

set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/homebrew/bin/llama-bench}"
N_TRIALS="${N_TRIALS:-3}"
NGEN="${NGEN:-256}"
PROMPT="${PROMPT:-Hello, my name is}"
THERMAL_SETTLE_SEC="${THERMAL_SETTLE_SEC:-60}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-bench}"
SKIP_THERMAL_GATE="${SKIP_THERMAL_GATE:-0}"
SKIP_RAM_GATE="${SKIP_RAM_GATE:-0}"
SKIP_HF2Q="${SKIP_HF2Q:-0}"
SKIP_LLAMA="${SKIP_LLAMA:-0}"

MODEL=""
LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$LABEL" ]]; then
  echo "ERROR: --model and --label are required" >&2
  echo "usage: $0 --model <gguf> --label <name>" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 2
fi
if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "ERROR: hf2q not found at $HF2Q_BIN" >&2
  exit 2
fi
if [[ ! -x "$LLAMA_BENCH_BIN" ]]; then
  echo "ERROR: llama-bench not found at $LLAMA_BENCH_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
META_OUT="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.metadata.json"
SUMMARY_OUT="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.summary.txt"

# ----------------------------------------------------------------------------
# RAM headroom gate (per `feedback_check_ram_before_inference`)
# ----------------------------------------------------------------------------
if [[ "$SKIP_RAM_GATE" != "1" ]]; then
  PAGE_SIZE_BYTES="$(vm_stat | head -1 | awk -F'of ' '{print $2}' | tr -d ' bytes)')"
  FREE_PAGES="$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')"
  INACTIVE_PAGES="$(vm_stat | awk '/Pages inactive/ {print $3}' | tr -d '.')"
  SPEC_PAGES="$(vm_stat | awk '/Pages speculative/ {print $3}' | tr -d '.')"
  TOTAL_AVAIL_BYTES=$(( ($FREE_PAGES + $INACTIVE_PAGES + $SPEC_PAGES) * $PAGE_SIZE_BYTES ))
  TOTAL_AVAIL_GB=$(( $TOTAL_AVAIL_BYTES / 1024 / 1024 / 1024 ))
  echo "RAM headroom: ${TOTAL_AVAIL_GB} GB available (need ≥ ${MIN_FREE_GB} GB)"
  if [[ $TOTAL_AVAIL_GB -lt $MIN_FREE_GB ]]; then
    echo "ERROR: insufficient RAM (${TOTAL_AVAIL_GB} GB < ${MIN_FREE_GB} GB)." >&2
    echo "       Loading the model risks OOM (rebooted M5 Max twice already)." >&2
    echo "       Set SKIP_RAM_GATE=1 to override (NOT recommended)." >&2
    exit 3
  fi
fi

# ----------------------------------------------------------------------------
# Run-wide metadata
# ----------------------------------------------------------------------------
{
  echo "{"
  echo "  \"date_utc\": \"$DATE_TAG\","
  echo "  \"label\": \"$LABEL\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"hostname\": \"$(hostname -s)\","
  echo "  \"macos_version\": \"$(sw_vers -productVersion)\","
  echo "  \"macos_build\": \"$(sw_vers -buildVersion)\","
  echo "  \"chip\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
  echo "  \"arch\": \"$(uname -m)\","
  echo "  \"darwin_kernel\": \"$(uname -r)\","
  echo "  \"hf2q_bin\": \"$HF2Q_BIN\","
  echo "  \"llama_bench_bin\": \"$LLAMA_BENCH_BIN\","
  echo "  \"prompt\": \"$PROMPT\","
  echo "  \"n_gen\": $NGEN,"
  echo "  \"n_trials\": $N_TRIALS,"
  echo "  \"thermal_settle_sec\": $THERMAL_SETTLE_SEC,"
  echo "  \"min_free_gb\": $MIN_FREE_GB,"
  echo "  \"hf2q_git_head\": \"$(git -C /opt/hf2q rev-parse HEAD)\","
  echo "  \"mlx_native_git_head\": \"$(git -C /opt/mlx-native rev-parse HEAD)\""
  echo "}"
} > "$META_OUT"
echo "wrote run metadata: $META_OUT"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
thermal_gate() {
  if [[ "$SKIP_THERMAL_GATE" == "1" ]]; then return 0; fi
  local therm
  therm="$(pmset -g therm 2>&1 || true)"
  if echo "$therm" | grep -q "CPU_Speed_Limit"; then
    local limit
    limit="$(echo "$therm" | awk -F'=' '/CPU_Speed_Limit/ {print $2}' | tr -d ' ')"
    if [[ -n "$limit" && "$limit" != "100" ]]; then
      echo "ERROR: thermal throttle detected (CPU_Speed_Limit=$limit). Aborting." >&2
      exit 2
    fi
  fi
}

# Extract decode tok/s from hf2q --benchmark stdout.
#
# Observed format (single-run --benchmark, per
# `reference_decode_benchmark_methodology`):
#
#   === Benchmark Results ===
#   Hardware: Apple M5 Max, 128 GB
#   Model: <path>
#   Prompt tokens: 18
#   Generated tokens: 256
#   Decode tok/s: 99.99
#
# Primary regex: "Decode tok/s: <X>".  Fallback: any "<N> tok/s" in
# stdout *or* stderr (mlx-native logs the per-decode tok/s line on
# stderr: "--- mlx-native: 256 tokens in <X>s (<Y> tok/s) ---").
extract_hf2q_tps() {
  local stdout_file="$1"
  local stderr_file="${stdout_file%.stdout}.stderr"
  local val
  val="$(awk -F': *' '/^Decode tok\/s:/ {print $2; exit}' "$stdout_file" \
            | tr -d ' \t')"
  if [[ -z "$val" ]]; then
    val="$(grep -Eo '[0-9]+\.[0-9]+ ?tok/s' "$stdout_file" 2>/dev/null \
              | head -1 \
              | grep -Eo '[0-9]+\.[0-9]+' | head -1)"
  fi
  if [[ -z "$val" && -f "$stderr_file" ]]; then
    val="$(grep -Eo '[0-9]+\.[0-9]+ ?tok/s' "$stderr_file" 2>/dev/null \
              | head -1 \
              | grep -Eo '[0-9]+\.[0-9]+' | head -1)"
  fi
  echo "${val:-0.0}"
}

# Extract decode tok/s from llama-bench output.
#
# Table row format (llama-bench tg<N>):
#   | model | size | params | backend | threads | test  | t/s              |
#   | ...   | ...  | ...    | ...     | ...     | tg256 | 103.96 ± 1.45    |
#
# We need the median (left of `±`), NOT the std-dev (right of `±`).
# Grab the first decimal that immediately precedes ` ± `.
extract_llama_tps() {
  local stdout_file="$1"
  local val
  # Primary: in the tg<NGEN> row, the value left of ` ± ` is the median.
  val="$(awk -v ngen="tg$NGEN" '
    $0 ~ ngen {
      n = split($0, parts, /\| */)
      for (i = 1; i <= n; i++) {
        if (match(parts[i], /[0-9]+\.[0-9]+ *± *[0-9]+\.[0-9]+/)) {
          v = substr(parts[i], RSTART, RLENGTH)
          sub(/ *±.*/, "", v)
          print v
          exit
        }
      }
    }
  ' "$stdout_file")"
  if [[ -z "$val" ]]; then
    # Fallback: any "<X> ± <Y>" anywhere; take the X half of the first.
    val="$(grep -Eo '[0-9]+\.[0-9]+ *± *[0-9]+\.[0-9]+' "$stdout_file" \
              | head -1 | awk -F' *± *' '{print $1}')"
  fi
  echo "${val:-0.0}"
}

# Run a single hf2q cold trial.
run_hf2q_trial() {
  local trial="$1"
  local stdout_out="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.hf2q.trial-${trial}.stdout"
  local stderr_out="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.hf2q.trial-${trial}.stderr"
  # Progress messages to STDERR — function's stdout must contain ONLY
  # the parsed tok/s value, since the caller captures it via $().
  echo "  hf2q trial $trial → $stdout_out" >&2
  set +e
  "$HF2Q_BIN" generate \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$NGEN" \
    --temperature 0 \
    --benchmark \
      > "$stdout_out" 2> "$stderr_out"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "    WARN: hf2q exit $rc; see $stderr_out" >&2
  fi
  extract_hf2q_tps "$stdout_out"
}

# Run a single llama-bench cold trial.
run_llama_trial() {
  local trial="$1"
  local stdout_out="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.llama.trial-${trial}.stdout"
  local stderr_out="$OUT_DIR/baseline-${LABEL}-${DATE_TAG}.llama.trial-${trial}.stderr"
  # Progress messages to STDERR — function's stdout must contain ONLY
  # the parsed tok/s value (caller captures via $()).
  echo "  llama-bench trial $trial → $stdout_out" >&2
  set +e
  "$LLAMA_BENCH_BIN" \
    -m "$MODEL" \
    -p 0 \
    -n "$NGEN" \
    -r 3 \
      > "$stdout_out" 2> "$stderr_out"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "    WARN: llama-bench exit $rc; see $stderr_out" >&2
  fi
  extract_llama_tps "$stdout_out"
}

# Compute median from a list of decimal values (sorted, pick middle).
median() {
  python3 -c "
import statistics, sys
vals = [float(x) for x in sys.argv[1:] if x not in ('', '0.0')]
if not vals:
    print('NaN')
else:
    print(f'{statistics.median(vals):.3f}')
" "$@"
}

# ----------------------------------------------------------------------------
# Trial loop
# ----------------------------------------------------------------------------
HF2Q_VALS=()
LLAMA_VALS=()

if [[ "$SKIP_HF2Q" != "1" ]]; then
  echo
  echo "=== hf2q cold-process trials ==="
  for trial in $(seq 1 "$N_TRIALS"); do
    thermal_gate
    val="$(run_hf2q_trial "$trial")"
    HF2Q_VALS+=("$val")
    echo "    trial $trial: $val tok/s"
    if [[ $trial -lt $N_TRIALS ]]; then
      echo "    settle ${THERMAL_SETTLE_SEC}s"
      sleep "$THERMAL_SETTLE_SEC"
    fi
  done
fi

if [[ "$SKIP_LLAMA" != "1" ]]; then
  echo
  echo "=== inter-side settle ${THERMAL_SETTLE_SEC}s ==="
  sleep "$THERMAL_SETTLE_SEC"

  echo
  echo "=== llama-bench cold-process trials ==="
  for trial in $(seq 1 "$N_TRIALS"); do
    thermal_gate
    val="$(run_llama_trial "$trial")"
    LLAMA_VALS+=("$val")
    echo "    trial $trial: $val tok/s"
    if [[ $trial -lt $N_TRIALS ]]; then
      echo "    settle ${THERMAL_SETTLE_SEC}s"
      sleep "$THERMAL_SETTLE_SEC"
    fi
  done
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
HF2Q_MEDIAN="$(median "${HF2Q_VALS[@]:-}")"
LLAMA_MEDIAN="$(median "${LLAMA_VALS[@]:-}")"
RATIO="$(python3 -c "
hf = float('$HF2Q_MEDIAN') if '$HF2Q_MEDIAN' != 'NaN' else 0.0
lc = float('$LLAMA_MEDIAN') if '$LLAMA_MEDIAN' != 'NaN' else 0.0
if lc > 0:
    print(f'{hf/lc:.4f}')
else:
    print('NaN')
")"

{
  echo "ADR-015 baseline — $LABEL"
  echo "date_utc:           $DATE_TAG"
  echo "model:              $MODEL"
  echo "hf2q_git_head:      $(git -C /opt/hf2q rev-parse --short HEAD)"
  echo "mlx_native_head:    $(git -C /opt/mlx-native rev-parse --short HEAD)"
  echo "n_gen:              $NGEN"
  echo "n_trials:           $N_TRIALS"
  echo "prompt:             $PROMPT"
  echo
  echo "hf2q tok/s (per trial):    ${HF2Q_VALS[@]:-N/A}"
  echo "hf2q tok/s (median):       $HF2Q_MEDIAN"
  echo "llama tok/s (per trial):   ${LLAMA_VALS[@]:-N/A}"
  echo "llama tok/s (median):      $LLAMA_MEDIAN"
  echo "ratio (hf2q / llama):      $RATIO"
  echo
  echo "ADR-015 D4 exit threshold: ratio ≥ 1.00× same-day median"
} | tee "$SUMMARY_OUT"

echo
echo "summary: $SUMMARY_OUT"
echo "metadata: $META_OUT"
