#!/usr/bin/env bash
#
# ADR-020 iter-19b step 2 — measure mean per-token KL divergence
# between the candidate model (DWQ Q4 from step 1) and the BF16
# baseline using the vendored `kld.py` (PR #1146).
#
# Per smcleod.net's published numbers (Apr 2026): mlx-community DWQ
# Q4 on Qwen 3.6 35B-A3B (the same model class as our reference) =
# mean per-token KLD 0.02663 vs an 8-bit reference.  Acceptance per
# ADR §8.2 row 19a: ≤ 0.030.

set -euo pipefail

cd "$(dirname "$0")"

CANDIDATE="${CANDIDATE:-./dwq_output}"
# NOTE: baseline must point at the local working dir (./abliterix_with_chat_template)
# not the raw HF id. mlx_lm.utils.load() fails on the upstream config's
# model_type="qwen3_5_moe_text" (mlx-lm 0.31.2 module is qwen3_5_moe). The working
# dir's config.json has the type patched and chat_template injected.
BASELINE="${BASELINE:-./abliterix_with_chat_template}"
TOP_K="${TOP_K:-1024}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1025}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-123}"
DATA_PATH="${DATA_PATH:-allenai/tulu-3-sft-mixture}"

echo "=== ADR-020 iter-19b step 02: mlx_lm.kld (vendored) ==="
echo "  candidate: $CANDIDATE"
echo "  baseline:  $BASELINE"
echo "  top_k=$TOP_K seq_len=$SEQUENCE_LENGTH num_samples=$NUM_SAMPLES"
echo ""

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_S=$(date +%s)

python3 kld.py \
    --model "$CANDIDATE" \
    --baseline-model "$BASELINE" \
    --top-k "$TOP_K" \
    --data-path "$DATA_PATH" \
    --sequence-length "$SEQUENCE_LENGTH" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    2>&1 | tee kld_run.log

END_S=$(date +%s)
DURATION=$((END_S - START_S))

# kld.py prints summary lines like
#   "Mean KLD: 0.026630"
# Capture the final mean.
MEAN_KLD=$(grep -E "Mean KLD" kld_run.log | tail -1 | sed -E 's/.*Mean KLD:?\s*([0-9.eE+-]+).*/\1/' || echo "")

echo ""
echo "=== Step 02 complete ==="
echo "  duration: ${DURATION}s"
echo "  mean KLD: $MEAN_KLD"
echo ""
echo "Acceptance gate (per ADR §8.2 row 19a):"
echo "  ≤ 0.030  → PASS  (matches mlx-lm published 0.02663 + 13% margin)"
echo "  > 0.100  → FAIL  (broken per smcleod's published threshold)"
echo ""

cat >> results.jsonl <<EOF
{"ts": "$START_TS", "step": "02_kld", "candidate": "$CANDIDATE", "baseline": "$BASELINE", "top_k": $TOP_K, "num_samples": $NUM_SAMPLES, "duration_sec": $DURATION, "mean_kld": "$MEAN_KLD"}
EOF
