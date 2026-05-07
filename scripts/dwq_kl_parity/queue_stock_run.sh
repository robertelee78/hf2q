#!/usr/bin/env bash
#
# ADR-020 iter-19b — sequenced post-training measurement chain.
#
# Step A (gated on current training PID 51393): wait for it to exit.
# Step B: kld.py on the abliterated DWQ output → row 19b "negative result"
#         number for the abliterated checkpoint.
# Step C: kld.py on the abliterated RTN-Q4 output → confirms whether the
#         already-low 0.050 valid-loss generalizes to kld.py's calibration
#         corpus (calibration_v5.txt vs validate()'s tulu-3 sample).
# Step D: kld.py on the abliterated DWQ output vs *teacher logit indices*
#         is implicit — kld.py uses the candidate model's tokenizer.
#
# Step E (gated on download PID): wait for the stock 35B-A3B BF16 to
#         finish downloading (~70 GB).
# Step F: build stock_working_dir/ analogous to abliterix_with_chat_template/
#         — symlinks + chat_template + model_type patch (only if needed).
# Step G: run 3-pass recipe on stock (RTN → targets → training).
# Step H: kld.py on stock DWQ → **load-bearing v1 parity number**.
# Step I: kld.py on stock RTN-Q4 → sanity vs smcleod's published 0.07418.
#
# All steps idempotent — re-runs skip already-completed work.
# Output JSONL: results.jsonl. Output log: queue_stock.log.

set -euo pipefail

cd "$(dirname "$0")"

LOG="queue_stock.log"
TRAIN_PID="${TRAIN_PID:-51393}"
DOWNLOAD_PID="${DOWNLOAD_PID:-}"
STOCK_REPO="Qwen/Qwen3.6-35B-A3B"
STOCK_LOCAL="./stock_working_dir"
STOCK_RTN="./stock_rtn_output"
STOCK_TARGETS="./stock_dwq_targets"
STOCK_DWQ="./stock_dwq_output"

# kld.py at line 328 hard-rejects baselines whose safetensors metadata
# `format` != 'mlx'.  Native HF safetensors have format='pt'.  We must
# convert BF16-as-mlx (--dtype bfloat16, no -q) before running kld.py.
ABLITERATED_MLX_BF16="./abliterated_mlx_bf16"
STOCK_MLX_BF16="./stock_mlx_bf16"

# Same hyperparameters as the abliterated run (canonical mlx-lm defaults).
BITS=4
GROUP_SIZE=64
NUM_SAMPLES=2048
MAX_SEQ_LENGTH=1025
LR=1e-6
BATCH_SIZE=2
SEED=123

log() {
    echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"
}

# ─────────────────────────────────────────────────────────────────────
# Step A — wait for current training to exit
# ─────────────────────────────────────────────────────────────────────
log "=== Step A: waiting for training PID $TRAIN_PID ==="
if ps -p "$TRAIN_PID" >/dev/null 2>&1; then
    while ps -p "$TRAIN_PID" >/dev/null 2>&1; do
        sleep 30
    done
    log "PID $TRAIN_PID exited"
else
    log "PID $TRAIN_PID already gone — proceeding"
fi

# Sanity: dwq_output should now exist
if [ ! -d "./dwq_output" ] || ! ls ./dwq_output/model-*.safetensors >/dev/null 2>&1; then
    log "WARN: ./dwq_output missing — abliterated training may have failed.  Continuing to step E (stock run)."
    SKIP_ABLITERATED=1
else
    SKIP_ABLITERATED=0
fi

# ─────────────────────────────────────────────────────────────────────
# Step B0 — convert abliterated BF16 to mlx-format (kld.py prerequisite)
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "$ABLITERATED_MLX_BF16/config.json" ] || ! ls "$ABLITERATED_MLX_BF16"/model-*.safetensors >/dev/null 2>&1; then
    log "=== Step B0: mlx_lm.convert (BF16-as-mlx) → $ABLITERATED_MLX_BF16 ==="
    START_S=$(date +%s)
    mlx_lm.convert --hf-path ./abliterix_with_chat_template \
        --mlx-path "$ABLITERATED_MLX_BF16" \
        --dtype bfloat16 \
        2>&1 | tee -a "$LOG"
    DUR=$(($(date +%s) - START_S))
    log "Step B0 done in ${DUR}s"
fi

# ─────────────────────────────────────────────────────────────────────
# Step B — kld.py: abliterated DWQ output vs abliterated BF16 reference
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_ABLITERATED" = "0" ] && [ ! -f "abliterated_dwq_kld.txt" ]; then
    log "=== Step B: kld.py on abliterated DWQ output ==="
    START_S=$(date +%s)
    python3 kld.py \
        --model ./dwq_output \
        --baseline-model "$ABLITERATED_MLX_BF16" \
        --top-k 1024 \
        --data-path allenai/tulu-3-sft-mixture \
        --sequence-length 1025 \
        --num-samples 512 \
        --batch-size 4 \
        --seed 123 \
        2>&1 | tee -a "$LOG"
    DUR=$(($(date +%s) - START_S))
    log "Step B done in ${DUR}s"
    grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > abliterated_dwq_kld.txt || true
fi

# ─────────────────────────────────────────────────────────────────────
# Step C — kld.py: abliterated RTN-Q4 output vs abliterated BF16
# ─────────────────────────────────────────────────────────────────────
if [ -d "./rtn_output" ] && [ ! -f "abliterated_rtn_kld.txt" ]; then
    log "=== Step C: kld.py on abliterated RTN-Q4 ==="
    START_S=$(date +%s)
    python3 kld.py \
        --model ./rtn_output \
        --baseline-model "$ABLITERATED_MLX_BF16" \
        --top-k 1024 \
        --data-path allenai/tulu-3-sft-mixture \
        --sequence-length 1025 \
        --num-samples 512 \
        --batch-size 4 \
        --seed 123 \
        2>&1 | tee -a "$LOG"
    DUR=$(($(date +%s) - START_S))
    log "Step C done in ${DUR}s"
    grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > abliterated_rtn_kld.txt || true
fi

# ─────────────────────────────────────────────────────────────────────
# Step E — wait for stock model download
# ─────────────────────────────────────────────────────────────────────
log "=== Step E: waiting for stock model download ==="
# Wait for the BF16 weights to appear.  Read the expected shard count
# from model.safetensors.index.json once it materializes (Qwen uses 26
# shards, abliterix uses 42 — don't hard-code).
HF_HUB_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
while true; do
    if [ -d "$HF_HUB_DIR" ]; then
        snap=$(ls -1 "$HF_HUB_DIR" 2>/dev/null | head -1)
        if [ -n "$snap" ]; then
            idx="$HF_HUB_DIR/$snap/model.safetensors.index.json"
            if [ -f "$idx" ]; then
                # Pull all unique shard filenames from weight_map values
                expected=$(python3 -c "
import json, sys
m = json.load(open('$idx'))
shards = set(m.get('weight_map', {}).values())
print(len(shards))
" 2>/dev/null || echo "0")
                count=$(ls -1 "$HF_HUB_DIR/$snap"/model-*.safetensors 2>/dev/null | wc -l)
                if [ "$expected" -gt 0 ] && [ "$count" -ge "$expected" ]; then
                    log "Stock download complete: $count / $expected shards in $snap"
                    STOCK_HF_DIR="$HF_HUB_DIR/$snap"
                    break
                fi
                log "Stock download progress: $count / $expected shards"
            else
                log "Stock download progress: index.json not yet present"
            fi
        fi
    fi
    sleep 60
done

# ─────────────────────────────────────────────────────────────────────
# Step F — build stock working dir
# ─────────────────────────────────────────────────────────────────────
if [ ! -d "$STOCK_LOCAL" ] || ! ls "$STOCK_LOCAL"/model-*.safetensors >/dev/null 2>&1; then
    log "=== Step F: build $STOCK_LOCAL ==="
    mkdir -p "$STOCK_LOCAL"
    # Symlink all source files
    for f in "$STOCK_HF_DIR"/*; do
        b=$(basename "$f")
        ln -sf "$f" "$STOCK_LOCAL/$b" 2>/dev/null || true
    done
    log "Built $STOCK_LOCAL with $(ls -1 "$STOCK_LOCAL" | wc -l) entries"

    # Patch model_type if upstream still has the qwen3_5_moe_text issue
    cfg="$STOCK_LOCAL/config.json"
    if [ -L "$cfg" ]; then
        # Materialize so we can edit
        cp -L "$cfg" "$cfg.tmp"
        mv "$cfg.tmp" "$cfg"
    fi
    if grep -q '"model_type"\s*:\s*"qwen3_5_moe_text"' "$cfg" 2>/dev/null; then
        log "Patching model_type qwen3_5_moe_text → qwen3_5_moe in $cfg"
        sed -i.bak 's/"qwen3_5_moe_text"/"qwen3_5_moe"/' "$cfg"
    else
        log "model_type already compatible — no patch needed"
    fi

    # Inject chat_template if missing (stock Qwen probably has it)
    tcfg="$STOCK_LOCAL/tokenizer_config.json"
    if [ -L "$tcfg" ]; then
        cp -L "$tcfg" "$tcfg.tmp"
        mv "$tcfg.tmp" "$tcfg"
    fi
    if ! grep -q '"chat_template"' "$tcfg" 2>/dev/null; then
        log "WARN: chat_template missing in stock $tcfg — copying from cached Qwen3.6-27B"
        SRC=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*/tokenizer_config.json 2>/dev/null | head -1)
        if [ -n "$SRC" ]; then
            python3 -c "
import json
src = json.load(open('$SRC'))
dst = json.load(open('$tcfg'))
if 'chat_template' in src:
    dst['chat_template'] = src['chat_template']
    json.dump(dst, open('$tcfg', 'w'), indent=2)
    print(f'Injected chat_template: {len(src[\"chat_template\"])} chars')
else:
    print('Source has no chat_template either — punting')
"
        fi
    else
        log "chat_template present — no injection needed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step F0 — convert stock BF16 to mlx-format (kld.py prerequisite for H/I)
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "$STOCK_MLX_BF16/config.json" ] || ! ls "$STOCK_MLX_BF16"/model-*.safetensors >/dev/null 2>&1; then
    log "=== Step F0: mlx_lm.convert (BF16-as-mlx) → $STOCK_MLX_BF16 ==="
    START_S=$(date +%s)
    mlx_lm.convert --hf-path "$STOCK_LOCAL" \
        --mlx-path "$STOCK_MLX_BF16" \
        --dtype bfloat16 \
        2>&1 | tee -a "$LOG"
    DUR=$(($(date +%s) - START_S))
    log "Step F0 done in ${DUR}s"
fi

# ─────────────────────────────────────────────────────────────────────
# Step G — 3-pass recipe on stock model
# ─────────────────────────────────────────────────────────────────────
log "=== Step G: 3-pass recipe on stock $STOCK_REPO ==="
START_S=$(date +%s)

# Pass 1: RTN-Q4 student
if [ -f "$STOCK_RTN/config.json" ] && ls "$STOCK_RTN"/model-*.safetensors >/dev/null 2>&1; then
    log "Pass 1: SKIP (RTN already at $STOCK_RTN)"
else
    log "Pass 1: mlx_lm.convert → $STOCK_RTN"
    mlx_lm.convert --hf-path "$STOCK_LOCAL" --mlx-path "$STOCK_RTN" \
        -q --q-bits "$BITS" --q-group-size "$GROUP_SIZE" --q-mode affine \
        2>&1 | tee -a "$LOG"
fi

# Pass 2: targets-only
if [ -d "$STOCK_TARGETS/train" ] && [ -d "$STOCK_TARGETS/valid" ] && \
   [ "$(ls -1 "$STOCK_TARGETS/train" 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Pass 2: SKIP (targets already at $STOCK_TARGETS)"
else
    log "Pass 2: mlx_lm.dwq --targets-only → $STOCK_TARGETS"
    mlx_lm.dwq --model "$STOCK_LOCAL" --target-dir "$STOCK_TARGETS" --targets-only \
        --bits "$BITS" --group-size "$GROUP_SIZE" \
        --num-samples "$NUM_SAMPLES" --max-seq-length "$MAX_SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" --seed "$SEED" \
        2>&1 | tee -a "$LOG"
fi

# Pass 3: training
log "Pass 3: mlx_lm.dwq training (Q4 student + cached targets + grad-checkpoint)"
mlx_lm.dwq \
    --model "$STOCK_LOCAL" \
    --quantized-model "$STOCK_RTN" \
    --target-dir "$STOCK_TARGETS" \
    --mlx-path "$STOCK_DWQ" \
    --bits "$BITS" --group-size "$GROUP_SIZE" \
    --num-samples "$NUM_SAMPLES" --max-seq-length "$MAX_SEQ_LENGTH" \
    --learning-rate "$LR" --batch-size "$BATCH_SIZE" --seed "$SEED" \
    --grad-checkpoint \
    2>&1 | tee -a "$LOG"

DUR=$(($(date +%s) - START_S))
log "Step G done in ${DUR}s"

# ─────────────────────────────────────────────────────────────────────
# Step H — LOAD-BEARING v1 parity number: kld.py on stock DWQ output
# ─────────────────────────────────────────────────────────────────────
log "=== Step H: LOAD-BEARING v1 parity — kld.py on stock DWQ ==="
START_S=$(date +%s)
python3 kld.py \
    --model "$STOCK_DWQ" \
    --baseline-model "$STOCK_MLX_BF16" \
    --top-k 1024 \
    --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 \
    --num-samples 512 \
    --batch-size 4 \
    --seed 123 \
    2>&1 | tee -a "$LOG"
DUR=$(($(date +%s) - START_S))
log "Step H done in ${DUR}s"
grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > stock_dwq_kld.txt || true

# ─────────────────────────────────────────────────────────────────────
# Step I — sanity baseline: kld.py on stock RTN-Q4
# ─────────────────────────────────────────────────────────────────────
log "=== Step I: sanity — kld.py on stock RTN-Q4 (expect ~0.07418) ==="
START_S=$(date +%s)
python3 kld.py \
    --model "$STOCK_RTN" \
    --baseline-model "$STOCK_MLX_BF16" \
    --top-k 1024 \
    --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 \
    --num-samples 512 \
    --batch-size 4 \
    --seed 123 \
    2>&1 | tee -a "$LOG"
DUR=$(($(date +%s) - START_S))
log "Step I done in ${DUR}s"
grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > stock_rtn_kld.txt || true

# ═════════════════════════════════════════════════════════════════════
# ADR-020 iter-19d enhancements — three additional measurements:
#   J: dwq_v2 (selective unfreeze + LR schedule + grad clip) on abliterated
#   K: dwq_v2 (selective unfreeze + LR schedule + grad clip) on stock
#   L: hf2q convert --quant imatrix-adaptive on abliterated, kld measure
#   M: hf2q convert --quant imatrix-adaptive on stock, kld measure
# ═════════════════════════════════════════════════════════════════════

ABL_DWQ_V2="./abliterated_dwq_v2_output"
STOCK_DWQ_V2="./stock_dwq_v2_output"
ABL_IMATRIX="./abliterated_imatrix_adaptive.gguf"
STOCK_IMATRIX="./stock_imatrix_adaptive.gguf"

# ─────────────────────────────────────────────────────────────────────
# Step J — dwq_v2 (sensitivity-gated + scheduled) on abliterated
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "abliterated_dwq_v2_kld.txt" ] && [ -d "./dwq_targets" ]; then
    if [ ! -d "$ABL_DWQ_V2" ] || ! ls "$ABL_DWQ_V2"/model-*.safetensors >/dev/null 2>&1; then
        log "=== Step J.1: dwq_v2 training on abliterated (sensitivity p=50, warmup=100, clip=1.0) ==="
        START_S=$(date +%s)
        HF2Q_DWQ_SENSITIVITY_PERCENTILE=50 \
        HF2Q_DWQ_WARMUP_STEPS=100 \
        HF2Q_DWQ_LR_FINAL_FRAC=0.1 \
        HF2Q_DWQ_GRAD_CLIP=1.0 \
        python3 dwq_v2.py \
            --model ./abliterix_with_chat_template \
            --quantized-model ./rtn_output \
            --target-dir ./dwq_targets \
            --mlx-path "$ABL_DWQ_V2" \
            --bits "$BITS" --group-size "$GROUP_SIZE" \
            --num-samples "$NUM_SAMPLES" --max-seq-length "$MAX_SEQ_LENGTH" \
            --learning-rate "$LR" --batch-size "$BATCH_SIZE" --seed "$SEED" \
            --grad-checkpoint \
            2>&1 | tee -a "$LOG"
        log "Step J.1 done in $(($(date +%s) - START_S))s"
    fi
    log "=== Step J.2: kld.py on abliterated dwq_v2 output ==="
    START_S=$(date +%s)
    python3 kld.py --model "$ABL_DWQ_V2" --baseline-model "$ABLITERATED_MLX_BF16" \
        --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
        --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123 \
        2>&1 | tee -a "$LOG"
    log "Step J.2 done in $(($(date +%s) - START_S))s"
    grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > abliterated_dwq_v2_kld.txt || true
fi

# ─────────────────────────────────────────────────────────────────────
# Step K — dwq_v2 on stock (gated on stock targets+rtn from earlier steps)
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "stock_dwq_v2_kld.txt" ] && [ -d "$STOCK_TARGETS" ] && [ -d "$STOCK_RTN" ]; then
    if [ ! -d "$STOCK_DWQ_V2" ] || ! ls "$STOCK_DWQ_V2"/model-*.safetensors >/dev/null 2>&1; then
        log "=== Step K.1: dwq_v2 training on stock (sensitivity p=50, warmup=100, clip=1.0) ==="
        START_S=$(date +%s)
        HF2Q_DWQ_SENSITIVITY_PERCENTILE=50 \
        HF2Q_DWQ_WARMUP_STEPS=100 \
        HF2Q_DWQ_LR_FINAL_FRAC=0.1 \
        HF2Q_DWQ_GRAD_CLIP=1.0 \
        python3 dwq_v2.py \
            --model "$STOCK_LOCAL" \
            --quantized-model "$STOCK_RTN" \
            --target-dir "$STOCK_TARGETS" \
            --mlx-path "$STOCK_DWQ_V2" \
            --bits "$BITS" --group-size "$GROUP_SIZE" \
            --num-samples "$NUM_SAMPLES" --max-seq-length "$MAX_SEQ_LENGTH" \
            --learning-rate "$LR" --batch-size "$BATCH_SIZE" --seed "$SEED" \
            --grad-checkpoint \
            2>&1 | tee -a "$LOG"
        log "Step K.1 done in $(($(date +%s) - START_S))s"
    fi
    log "=== Step K.2: kld.py on stock dwq_v2 output ==="
    START_S=$(date +%s)
    python3 kld.py --model "$STOCK_DWQ_V2" --baseline-model "$STOCK_MLX_BF16" \
        --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
        --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123 \
        2>&1 | tee -a "$LOG"
    log "Step K.2 done in $(($(date +%s) - START_S))s"
    grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' "$LOG" | tail -1 > stock_dwq_v2_kld.txt || true
fi

# ─────────────────────────────────────────────────────────────────────
# Step L — hf2q convert --quant imatrix-adaptive on abliterated + kld
# ─────────────────────────────────────────────────────────────────────
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q-adr020-iter10/target/release/hf2q}"
if [ ! -f "abliterated_imatrix_kld.txt" ]; then
    if [ ! -f "$HF2Q_BIN" ]; then
        log "Step L: SKIP — hf2q binary not built at $HF2Q_BIN (run 'cargo build --release' separately)"
    else
        if [ ! -f "$ABL_IMATRIX" ]; then
            log "=== Step L.1: hf2q convert imatrix-adaptive on abliterated ==="
            START_S=$(date +%s)
            "$HF2Q_BIN" convert --hf-path ./abliterix_with_chat_template \
                --output "$ABL_IMATRIX" \
                --quant imatrix-adaptive \
                2>&1 | tee -a "$LOG" || log "Step L.1 hf2q convert failed — continuing"
            log "Step L.1 done in $(($(date +%s) - START_S))s"
        fi
        # kld.py expects mlx-format, not GGUF — skip the kld step for now
        # and just log that the imatrix conversion landed.
        log "Step L.2: kld.py on imatrix-adaptive GGUF requires GGUF→mlx bridge — DEFERRED"
        log "  (record GGUF size + tensor-bit allocation as the L outputs)"
        ls -la "$ABL_IMATRIX" 2>&1 | tee -a "$LOG"
        echo "imatrix-adaptive-gguf-only" > abliterated_imatrix_kld.txt
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step M — hf2q convert --quant imatrix-adaptive on stock + kld
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "stock_imatrix_kld.txt" ]; then
    if [ ! -f "$HF2Q_BIN" ]; then
        log "Step M: SKIP — hf2q binary not built at $HF2Q_BIN"
    else
        if [ ! -f "$STOCK_IMATRIX" ]; then
            log "=== Step M.1: hf2q convert imatrix-adaptive on stock ==="
            START_S=$(date +%s)
            "$HF2Q_BIN" convert --hf-path "$STOCK_LOCAL" \
                --output "$STOCK_IMATRIX" \
                --quant imatrix-adaptive \
                2>&1 | tee -a "$LOG" || log "Step M.1 hf2q convert failed — continuing"
            log "Step M.1 done in $(($(date +%s) - START_S))s"
        fi
        log "Step M.2: kld.py on imatrix-adaptive GGUF requires GGUF→mlx bridge — DEFERRED"
        ls -la "$STOCK_IMATRIX" 2>&1 | tee -a "$LOG"
        echo "imatrix-adaptive-gguf-only" > stock_imatrix_kld.txt
    fi
fi

log "=== Queue complete — all steps ==="
log "Summary:"
log "  abliterated RTN-Q4 KL    : $(cat abliterated_rtn_kld.txt 2>/dev/null || echo 'n/a')"
log "  abliterated DWQ-Q4 KL    : $(cat abliterated_dwq_kld.txt 2>/dev/null || echo 'n/a')"
log "  abliterated DWQ-v2 KL    : $(cat abliterated_dwq_v2_kld.txt 2>/dev/null || echo 'n/a')"
log "  abliterated imatrix-adapt: $(cat abliterated_imatrix_kld.txt 2>/dev/null || echo 'n/a')"
log "  stock       RTN-Q4 KL    : $(cat stock_rtn_kld.txt 2>/dev/null || echo 'n/a')"
log "  stock       DWQ-Q4 KL    : $(cat stock_dwq_kld.txt 2>/dev/null || echo 'n/a')"
log "  stock       DWQ-v2 KL    : $(cat stock_dwq_v2_kld.txt 2>/dev/null || echo 'n/a')"
log "  stock       imatrix-adapt: $(cat stock_imatrix_kld.txt 2>/dev/null || echo 'n/a')"
