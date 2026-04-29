#!/usr/bin/env bash
# scripts/bench-matrix.sh
#
# ADR-015 iter18 — paired-binary cold-SoC bench across (model × quant) cells.
# Operationalizes feedback_speed_bar_full_matrix: hf2q ≥1.00× llama.cpp
# across ALL quants/conversions/lengths/modes on the same hardware.
#
# Output: a single markdown matrix at $OUT_DIR/matrix-${DATE}.md showing
# per-cell hf2q tok/s, llama tok/s, ratio, Δpp vs llama, and the per-trial
# values in a fenced detail block. The table is the deliverable.
#
# Per cell:
#   - bench-baseline.sh runs hf2q + llama-bench at NGEN=256, 3 cold trials,
#     prompt "Hello, my name is" (matches D4 reference). Per-trial gates
#     archived per the iter12 fix (pmset/vm_stat/brain-stat pre+post).
#
# Usage:
#   scripts/bench-matrix.sh                # all cells
#   CELLS="qwen3.6-35b-a3b-dwq46,gemma-26B-dwq" scripts/bench-matrix.sh
#
# Env:
#   N_TRIALS  (default 3)
#   NGEN      (default 256)
#   SETTLE_BETWEEN_CELLS_SEC (default 90)
#
# Rationale: feedback_speed_bar_full_matrix calls for ≥1.00× across the
# full grid; iter11-17 chased a single (model, quant) cell in a loop and
# all 13 falsifications were on that one cell. The user's iter18 directive
# is to stop making blind changes and instead "wire up a test harness that
# lets us know where we're slower" — across the full matrix, not one cell.

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-/opt/homebrew/bin/llama-bench}"
N_TRIALS="${N_TRIALS:-3}"
NGEN="${NGEN:-256}"
PROMPT="${PROMPT:-Hello, my name is}"
SETTLE_BETWEEN_CELLS_SEC="${SETTLE_BETWEEN_CELLS_SEC:-90}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter18/bench}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT_DIR"
MATRIX="$OUT_DIR/matrix-${DATE_TAG}.md"

# (cell_label, model_path, quant_signature)
declare -a CELLS_ALL=(
  "qwen3.6-35b-a3b-dwq46|/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf|MoE-35B-A3B × dwq46 (Q4 base / Q6 sensitive)"
  "qwen3.6-35b-a3b-dwq48|/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf|MoE-35B-A3B × dwq48 (Q4 base / Q8 sensitive)"
  "qwen3.6-35b-a3b-apex|/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf|MoE-35B-A3B × apex (Q8-ish baseline)"
  "qwen3.6-27b-dwq46|/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf|dense-27B × dwq46"
  "qwen3.6-27b-dwq48|/opt/hf2q/models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf|dense-27B × dwq48"
  "gemma-26B-dwq|/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf|gemma-26B × dwq"
)

# Optional CELLS env to subset by label
if [[ -n "${CELLS:-}" ]]; then
  declare -a CELLS_SUBSET=()
  IFS=',' read -ra WANTED <<< "${CELLS}"
  for entry in "${CELLS_ALL[@]}"; do
    label="${entry%%|*}"
    for w in "${WANTED[@]}"; do
      [[ "$label" == "$w" ]] && CELLS_SUBSET+=("$entry")
    done
  done
  CELLS_ALL=("${CELLS_SUBSET[@]}")
fi

# Pre-flight
[[ -x "$HF2Q_BIN" ]] || { echo "FAIL: HF2Q_BIN not executable: $HF2Q_BIN" >&2; exit 1; }
[[ -x "$LLAMA_BENCH_BIN" ]] || { echo "FAIL: LLAMA_BENCH_BIN not executable: $LLAMA_BENCH_BIN" >&2; exit 1; }
[[ -x /opt/hf2q/scripts/bench-baseline.sh ]] || { echo "FAIL: bench-baseline.sh not executable" >&2; exit 1; }

HF2Q_GIT_HEAD=$(git -C "$(dirname "$(dirname "$HF2Q_BIN")")" rev-parse HEAD 2>/dev/null || echo unknown)
MLX_GIT_HEAD=$(git -C /opt/mlx-native rev-parse HEAD 2>/dev/null || echo unknown)

echo "=== ADR-015 iter18 — bench-matrix run ${DATE_TAG} ==="
echo "hf2q HEAD : $HF2Q_GIT_HEAD"
echo "mlx HEAD  : $MLX_GIT_HEAD"
echo "n_trials  : $N_TRIALS"
echo "n_gen     : $NGEN"
echo "prompt    : $PROMPT"
echo "out       : $OUT_DIR"
echo "matrix    : $MATRIX"
echo "cells     : ${#CELLS_ALL[@]}"
for c in "${CELLS_ALL[@]}"; do echo "  - ${c%%|*}"; done
echo

{
  echo "# ADR-015 iter18 bench-matrix"
  echo
  echo "**Run UTC:** $DATE_TAG  "
  echo "**hf2q HEAD:** \`$HF2Q_GIT_HEAD\`  "
  echo "**mlx-native HEAD:** \`$MLX_GIT_HEAD\`  "
  echo "**Trials per side:** $N_TRIALS  "
  echo "**NGEN:** $NGEN  "
  echo "**Prompt:** \`$PROMPT\`  "
  echo "**Cells:** ${#CELLS_ALL[@]}  "
  echo
  echo "## Matrix"
  echo
  echo "| cell | quant signature | hf2q t/s (median) | llama t/s (median) | ratio | Δpp vs 1.00× | hf2q per-trial | llama per-trial |"
  echo "|---|---|---:|---:|---:|---:|---|---|"
} > "$MATRIX"

cell_idx=0
for entry in "${CELLS_ALL[@]}"; do
  cell_idx=$((cell_idx + 1))
  IFS='|' read -r label model_path quant_sig <<< "$entry"

  if [[ ! -f "$model_path" ]]; then
    echo "[$cell_idx/${#CELLS_ALL[@]}] SKIP $label — model not found: $model_path" >&2
    echo "| $label | $quant_sig | MISSING | MISSING | — | — | — | — |" >> "$MATRIX"
    continue
  fi

  echo
  echo "[$cell_idx/${#CELLS_ALL[@]}] >>> $label"
  echo "    model: $model_path"
  echo "    quant: $quant_sig"

  cell_out_dir="$OUT_DIR/$label"
  mkdir -p "$cell_out_dir"

  if HF2Q_BIN="$HF2Q_BIN" \
     LLAMA_BENCH_BIN="$LLAMA_BENCH_BIN" \
     N_TRIALS="$N_TRIALS" \
     NGEN="$NGEN" \
     PROMPT="$PROMPT" \
     OUT_DIR="$cell_out_dir" \
     /opt/hf2q/scripts/bench-baseline.sh \
       --model "$model_path" \
       --label "$label" 2>&1 | tee "$cell_out_dir/run.log"; then

    summary=$(ls -t "$cell_out_dir"/*.summary.txt 2>/dev/null | head -1)
    if [[ -z "$summary" ]]; then
      echo "    WARN: no summary.txt for $label"
      echo "| $label | $quant_sig | ERR | ERR | — | — | — | — |" >> "$MATRIX"
      continue
    fi

    hf2q_per=$(grep "^hf2q tok/s (per trial):" "$summary" | sed 's/.*per trial)://' | xargs)
    hf2q_med=$(grep "^hf2q tok/s (median):" "$summary" | awk '{print $NF}')
    llama_per=$(grep "^llama tok/s (per trial):" "$summary" | sed 's/.*per trial)://' | xargs)
    llama_med=$(grep "^llama tok/s (median):" "$summary" | awk '{print $NF}')
    ratio=$(grep "^ratio (hf2q / llama):" "$summary" | awk '{print $NF}')

    if [[ -n "$ratio" && "$ratio" != "NaN" ]]; then
      delta_pp=$(awk -v r="$ratio" 'BEGIN { printf "%.2f", (r - 1.00) * 100 }')
    else
      delta_pp="—"
    fi

    echo "    hf2q  : $hf2q_med t/s ($hf2q_per)"
    echo "    llama : $llama_med t/s ($llama_per)"
    echo "    ratio : $ratio  ΔppFromUnity: $delta_pp"

    echo "| $label | $quant_sig | $hf2q_med | $llama_med | $ratio | $delta_pp | $hf2q_per | $llama_per |" >> "$MATRIX"
  else
    echo "    FAIL: bench-baseline.sh exited non-zero for $label"
    echo "| $label | $quant_sig | FAIL | FAIL | — | — | — | — |" >> "$MATRIX"
  fi

  if (( cell_idx < ${#CELLS_ALL[@]} )); then
    echo "    settle ${SETTLE_BETWEEN_CELLS_SEC}s between cells..."
    sleep "$SETTLE_BETWEEN_CELLS_SEC"
  fi
done

{
  echo
  echo "## Standing-pin reading"
  echo
  echo "Per \`feedback_speed_bar_full_matrix\`: hf2q must be ≥1.00× across all quants/conversions/lengths/modes on the same hardware. **Any cell with Δpp < 0** is a regression vs the standing bar; the matrix above is the durable evidence."
  echo
  echo "Same-day llama drift envelope: ≤ ±1pp per \`project_end_gate_reality_check\`."
  echo
  echo "## Cells artifacts"
  echo
  for c in "${CELLS_ALL[@]}"; do
    label="${c%%|*}"
    echo "- \`$OUT_DIR/$label/\` — per-cell summary, metadata, per-trial logs + gates"
  done
} >> "$MATRIX"

echo
echo "=== matrix written to: $MATRIX ==="
cat "$MATRIX"
