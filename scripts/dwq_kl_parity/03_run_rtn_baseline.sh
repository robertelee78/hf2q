#!/usr/bin/env bash
#
# ADR-020 iter-19b step 3 — measure mean per-token KLD for the naive
# RTN-Q4 (no DWQ training) variant against the BF16 baseline.  This
# confirms the canonical "DWQ ~2.8× better than RTN" relationship per
# smcleod.net's published numbers (Apr 2026): mlx-community 4bit
# (no DWQ) on Qwen 3.6 35B-A3B = 0.07418 vs DWQ 0.02663.
#
# Uses `mlx_lm.convert` for naive RTN, then runs the same kld.py
# harness as step 2.

set -euo pipefail

cd "$(dirname "$0")"

REF_MODEL="${REF_MODEL:-jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated}"
RTN_OUT="${RTN_OUT:-./rtn_q4_output}"
BITS="${BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"
TOP_K="${TOP_K:-1024}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1025}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-123}"
DATA_PATH="${DATA_PATH:-allenai/tulu-3-sft-mixture}"

echo "=== ADR-020 iter-19b step 03: RTN-Q4 baseline ==="
echo ""

# Step 3a — naive RTN Q4 via mlx_lm.convert.
if [ ! -d "$RTN_OUT" ]; then
    echo "[3a] producing RTN-Q4 via mlx_lm.convert ..."
    mlx_lm.convert \
        --hf-path "$REF_MODEL" \
        --mlx-path "$RTN_OUT" \
        --quantize \
        --q-bits "$BITS" \
        --q-group-size "$GROUP_SIZE" \
        2>&1 | tee rtn_convert.log
else
    echo "[3a] RTN output already exists at $RTN_OUT — skipping convert"
fi

# Step 3b — measure KLD.
echo "[3b] running kld.py against RTN-Q4 ..."
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_S=$(date +%s)

python3 kld.py \
    --model "$RTN_OUT" \
    --baseline-model "$REF_MODEL" \
    --top-k "$TOP_K" \
    --data-path "$DATA_PATH" \
    --sequence-length "$SEQUENCE_LENGTH" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    2>&1 | tee rtn_kld_run.log

END_S=$(date +%s)
DURATION=$((END_S - START_S))
MEAN_KLD=$(grep -E "Mean KLD" rtn_kld_run.log | tail -1 | sed -E 's/.*Mean KLD:?\s*([0-9.eE+-]+).*/\1/' || echo "")

echo ""
echo "=== Step 03 complete ==="
echo "  RTN-Q4 mean KLD: $MEAN_KLD"
echo ""
echo "Expected: ~0.074 per smcleod's published RTN-Q4 number on this model class."
echo "If DWQ (step 02) beat this by ≥2× we've reproduced mlx-lm's published improvement."
echo ""

cat >> results.jsonl <<EOF
{"ts": "$START_TS", "step": "03_rtn_baseline", "candidate": "$RTN_OUT", "baseline": "$REF_MODEL", "duration_sec": $DURATION, "mean_kld": "$MEAN_KLD"}
EOF
