#!/usr/bin/env bash
#
# ADR-020 iter-19b step 1 — produce a DWQ Q4 affine-quantized
# checkpoint from the BF16 reference using the canonical
# `mlx_lm.dwq` tool.
#
# 3-PASS RECIPE (canonical low-memory).  Without this split the
# default invocation peaks ~75 GB and OOMs on 128 GB unified memory
# at the first training step (the BF16 teacher and the Q4 student
# both live in memory plus Adam state plus activations).
#
# Pass 1: mlx_lm.convert  -- materialize an RTN Q4 student to disk
#                            (≈18 GB).  This is a side-product we
#                            also use as the RTN baseline in step 03.
# Pass 2: mlx_lm.dwq --targets-only --target-dir DIR
#                         -- teacher forward only; writes top-1024
#                            logits per batch to disk (~10 GB).  No
#                            Adam, no student, no autograd.
# Pass 3: mlx_lm.dwq --quantized-model RTN --target-dir DIR --grad-checkpoint
#                         -- training run loads ONLY the Q4 student
#                            (≈18 GB) + Adam state for trainable
#                            scales/biases (~12 GB).  Targets stream
#                            from disk.  Total peak ≈ 35-40 GB.
#
# Defaults match mlx-lm's published recipe:
#   bits=4, group_size=64, mode=affine (mlx-lm/quant/dwq.py defaults)
#   temperature=2.0, learning-rate=1e-6
#   num-samples=2048, max-seq-length=1025
#
# Override any of these via env vars; e.g. `BITS=4 GROUP_SIZE=64 ./01_run_dwq.sh`.

set -euo pipefail

cd "$(dirname "$0")"

REF_MODEL="${REF_MODEL:-./abliterix_with_chat_template}"
OUT_DIR="${OUT_DIR:-./dwq_output}"
RTN_DIR="${RTN_DIR:-./rtn_output}"
TARGET_DIR="${TARGET_DIR:-./dwq_targets}"
BITS="${BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-2048}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1025}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-123}"

mkdir -p "$OUT_DIR"

echo "=== ADR-020 iter-19b step 01: mlx_lm.dwq (3-pass low-memory recipe) ==="
echo "  reference:     $REF_MODEL"
echo "  output (DWQ):  $OUT_DIR"
echo "  output (RTN):  $RTN_DIR"
echo "  targets dir:   $TARGET_DIR"
echo "  bits=$BITS group_size=$GROUP_SIZE temp=2.0 lr=$LR"
echo "  num-samples=$NUM_SAMPLES max-seq-length=$MAX_SEQ_LENGTH batch=$BATCH_SIZE"
echo ""

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_S=$(date +%s)

# -----------------------------------------------------------------------------
# Pass 1: Pre-quantize the student via mlx_lm.convert (idempotent — skip if
# rtn_output already has weights).  Output is RTN Q4 affine bits/group_size.
# -----------------------------------------------------------------------------
if [ -f "$RTN_DIR/config.json" ] && ls "$RTN_DIR"/model-*.safetensors >/dev/null 2>&1; then
    echo "--- Pass 1: SKIP (RTN student already at $RTN_DIR)"
else
    echo "--- Pass 1: mlx_lm.convert → $RTN_DIR (RTN Q4)"
    PASS1_S=$(date +%s)
    mlx_lm.convert \
        --hf-path "$REF_MODEL" \
        --mlx-path "$RTN_DIR" \
        -q --q-bits "$BITS" --q-group-size "$GROUP_SIZE" --q-mode affine \
        2>&1 | tee -a dwq_run.log
    PASS1_DUR=$(($(date +%s) - PASS1_S))
    echo "--- Pass 1 done in ${PASS1_DUR}s"
fi
echo ""

# -----------------------------------------------------------------------------
# Pass 2: Pre-compute teacher targets to disk (top-1024 logits per batch).
# Skipped if target_dir/train and target_dir/valid already exist.
# -----------------------------------------------------------------------------
if [ -d "$TARGET_DIR/train" ] && [ -d "$TARGET_DIR/valid" ] && \
   [ "$(ls -1 "$TARGET_DIR/train" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "--- Pass 2: SKIP (targets already at $TARGET_DIR)"
else
    echo "--- Pass 2: mlx_lm.dwq --targets-only → $TARGET_DIR"
    PASS2_S=$(date +%s)
    mlx_lm.dwq \
        --model "$REF_MODEL" \
        --target-dir "$TARGET_DIR" \
        --targets-only \
        --bits "$BITS" \
        --group-size "$GROUP_SIZE" \
        --num-samples "$NUM_SAMPLES" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee -a dwq_run.log
    PASS2_DUR=$(($(date +%s) - PASS2_S))
    echo "--- Pass 2 done in ${PASS2_DUR}s"
fi
echo ""

# -----------------------------------------------------------------------------
# Pass 3: Training run with Q4 student + cached targets + grad-checkpoint.
# This is the canonical apples-to-apples comparable.
# -----------------------------------------------------------------------------
echo "--- Pass 3: mlx_lm.dwq --quantized-model + --target-dir + --grad-checkpoint"
PASS3_S=$(date +%s)
mlx_lm.dwq \
    --model "$REF_MODEL" \
    --quantized-model "$RTN_DIR" \
    --target-dir "$TARGET_DIR" \
    --mlx-path "$OUT_DIR" \
    --bits "$BITS" \
    --group-size "$GROUP_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --learning-rate "$LR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --grad-checkpoint \
    2>&1 | tee -a dwq_run.log
PASS3_DUR=$(($(date +%s) - PASS3_S))
echo "--- Pass 3 done in ${PASS3_DUR}s"

END_S=$(date +%s)
DURATION=$((END_S - START_S))

# mlx_lm.dwq prints "Validation: it=N, loss=L.LLL" lines; capture
# the first (=initial pre-training) and last (=final post-training).
INITIAL_LOSS=$(grep -E "^Validation: it=0," dwq_run.log | head -1 | sed -E 's/.*loss=([0-9.]+).*/\1/' || echo "")
FINAL_LOSS=$(grep -E "^Validation:" dwq_run.log | tail -1 | sed -E 's/.*loss=([0-9.]+).*/\1/' || echo "")

echo ""
echo "=== Step 01 complete (3 passes) ==="
echo "  total duration: ${DURATION}s"
echo "  initial valid:  $INITIAL_LOSS"
echo "  final valid:    $FINAL_LOSS"
echo ""

# Append JSON record
cat >> results.jsonl <<EOF
{"ts": "$START_TS", "step": "01_dwq_3pass", "ref_model": "$REF_MODEL", "rtn_dir": "$RTN_DIR", "target_dir": "$TARGET_DIR", "out_dir": "$OUT_DIR", "bits": $BITS, "group_size": $GROUP_SIZE, "num_samples": $NUM_SAMPLES, "max_seq_length": $MAX_SEQ_LENGTH, "batch_size": $BATCH_SIZE, "duration_sec": $DURATION, "valid_loss_initial": "$INITIAL_LOSS", "valid_loss_final": "$FINAL_LOSS"}
EOF
