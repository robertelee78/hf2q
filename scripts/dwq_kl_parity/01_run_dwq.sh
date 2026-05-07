#!/usr/bin/env bash
#
# ADR-020 iter-19b step 1 — produce a DWQ Q4 affine-quantized
# checkpoint from the BF16 reference using the canonical
# `mlx_lm.dwq` tool.
#
# Defaults match mlx-lm's published recipe:
#   bits=4, group_size=64, mode=affine (mlx-lm/quant/dwq.py defaults)
#   temperature=2.0, learning-rate=1e-6, batch-size=4
#   num-samples=2048, max-seq-length=1025
#
# Override any of these via env vars; e.g. `BITS=4 GROUP_SIZE=64 ./01_run_dwq.sh`.
#
# Runtime: ~hours on M5 Max for 35B-A3B MoE (most params don't
# activate per forward; output is ~18 GB).
# Memory: ~75 GB peak unified memory.

set -euo pipefail

cd "$(dirname "$0")"

REF_MODEL="${REF_MODEL:-jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated}"
OUT_DIR="${OUT_DIR:-./dwq_output}"
BITS="${BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-2048}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1025}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-123}"

mkdir -p "$OUT_DIR"

echo "=== ADR-020 iter-19b step 01: mlx_lm.dwq ==="
echo "  reference: $REF_MODEL"
echo "  output:    $OUT_DIR"
echo "  bits=$BITS group_size=$GROUP_SIZE temp=2.0 lr=$LR"
echo "  num-samples=$NUM_SAMPLES max-seq-length=$MAX_SEQ_LENGTH"
echo ""

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_S=$(date +%s)

mlx_lm.dwq \
    --model "$REF_MODEL" \
    --mlx-path "$OUT_DIR" \
    --bits "$BITS" \
    --group-size "$GROUP_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --learning-rate "$LR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    2>&1 | tee dwq_run.log

END_S=$(date +%s)
DURATION=$((END_S - START_S))

# mlx_lm.dwq prints "Validation: it=N, loss=L.LLL" lines; capture
# the first (=initial pre-training) and last (=final post-training).
INITIAL_LOSS=$(grep -E "^Validation: it=0," dwq_run.log | head -1 | sed -E 's/.*loss=([0-9.]+).*/\1/' || echo "")
FINAL_LOSS=$(grep -E "^Validation:" dwq_run.log | tail -1 | sed -E 's/.*loss=([0-9.]+).*/\1/' || echo "")

echo ""
echo "=== Step 01 complete ==="
echo "  duration:      ${DURATION}s"
echo "  initial valid: $INITIAL_LOSS"
echo "  final valid:   $FINAL_LOSS"
echo ""

# Append JSON record
cat >> results.jsonl <<EOF
{"ts": "$START_TS", "step": "01_dwq", "ref_model": "$REF_MODEL", "out_dir": "$OUT_DIR", "bits": $BITS, "group_size": $GROUP_SIZE, "num_samples": $NUM_SAMPLES, "duration_sec": $DURATION, "valid_loss_initial": "$INITIAL_LOSS", "valid_loss_final": "$FINAL_LOSS"}
EOF
