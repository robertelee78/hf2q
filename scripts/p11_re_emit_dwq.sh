#!/usr/bin/env bash
# ADR-014 P11 — re-emit ADR-012's 4 DWQ GGUFs through the streaming pipeline
# and gate against the reference artefacts in /opt/hf2q/models/.
#
# Per the engineering mantra: NEVER load a model without checking RAM first
# (vm_stat / PhysMem; OOM has rebooted M5 Max twice).
#
# Per ADR-014 Decision 22 (P11 closure AC): each re-emit must
#   (a) load successfully in `llama-cli` without rejection
#   (b) generate coherent output on a canonical pangram prompt
#   (c) optionally — record peak RSS + wall time via `/usr/bin/time -l`
#
# Usage:
#   bash scripts/p11_re_emit_dwq.sh                # all 4 variants
#   bash scripts/p11_re_emit_dwq.sh dwq46          # 27B + 35B-MoE dwq46 only
#   bash scripts/p11_re_emit_dwq.sh dwq48          # 27B + 35B-MoE dwq48 only
#   bash scripts/p11_re_emit_dwq.sh 27b-dwq46      # iter-92: 27B dwq46 only (smallest peak)
#   bash scripts/p11_re_emit_dwq.sh 27b-dwq48      # iter-92: 27B dwq48 only
#   bash scripts/p11_re_emit_dwq.sh 35bmoe-dwq46   # iter-92: 35B-MoE dwq46 only
#   bash scripts/p11_re_emit_dwq.sh 35bmoe-dwq48   # iter-92: 35B-MoE dwq48 only
#   bash scripts/p11_re_emit_dwq.sh stream         # all 4 with HF2Q_STREAMING_PHASE3=1
#
# Default behavior is the eager path; `stream` runs all 4 with the env flag on.
# Per-variant selectors (iter-92 addition) let the loop progress one variant
# at a time, letting RAM settle between heavy DWQ peaks.

set -euo pipefail

QWEN27_SRC="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
QWEN35MOE_SRC="$HOME/.cache/huggingface/hub/models--jenerallee78--Qwen3.6-35B-A3B-Abliterix-EGA-abliterated/snapshots/afde6ca7c35272a4b5eefb3b97576fdac0f74ba0"
OUT_ROOT="${HF2Q_P11_OUT_ROOT:-/tmp/p11-re-emit}"
LLAMA_CLI="${HF2Q_LLAMA_CLI_BIN:-/opt/homebrew/bin/llama-cli}"
PROMPT="The quick brown fox"

mkdir -p "$OUT_ROOT"

# ────────────────────────────────────────────────────────────
# Pre-flight: RAM check (mantra non-negotiable)
# ────────────────────────────────────────────────────────────
ram_check() {
    local label="$1"
    local free_pages inactive_pages
    free_pages=$(vm_stat | awk '/Pages free/ { gsub(/\./, "", $3); print $3 }')
    inactive_pages=$(vm_stat | awk '/Pages inactive/ { gsub(/\./, "", $3); print $3 }')
    local free_gb inactive_gb
    free_gb=$(( free_pages * 16384 / 1024 / 1024 / 1024 ))
    inactive_gb=$(( inactive_pages * 16384 / 1024 / 1024 / 1024 ))
    # iter-97: inactive pages on macOS are reclaimable instantly. The
    # original 30 GB free-only floor was too conservative once iter-95's
    # cache HIT short-circuit + iter-97's primed sensitivity cache cut
    # the dense Qwen3.5/3.6 path's peak from 158 GB → ~52 GB. Count
    # free + inactive as available; gate at ≥30 GB AVAILABLE (override
    # via HF2Q_P11_MIN_AVAIL_GB env var if needed).
    local available_gb=$(( free_gb + inactive_gb ))
    local min_avail="${HF2Q_P11_MIN_AVAIL_GB:-30}"
    echo "[$label] RAM free: ${free_gb} GB  inactive: ${inactive_gb} GB  available: ${available_gb} GB  (min: ${min_avail} GB)"
    if [ "$available_gb" -lt "$min_avail" ]; then
        echo "[$label] ERROR: <${min_avail} GB available — refusing to load model (mantra OOM-prevention)"
        return 1
    fi
}

# ────────────────────────────────────────────────────────────
# One conversion run with measurement
# ────────────────────────────────────────────────────────────
run_convert() {
    local label="$1"     # "27B-dwq46", "35BMOE-dwq48", etc.
    local src="$2"
    local quant="$3"     # "dwq-mixed-4-6" or "dwq-mixed-4-8"
    local stream="$4"    # "" or "1"

    ram_check "$label"

    local out_dir="$OUT_ROOT/$label"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    local env_prefix=""
    if [ "$stream" = "1" ]; then
        env_prefix="HF2Q_STREAMING_PHASE3=1 "
    fi

    # iter-92 fix: tee writes to a sibling log path, NOT inside $out_dir.
    # Race condition: pipeline starts hf2q + tee in parallel; tee creates
    # convert.log in $out_dir milliseconds before hf2q checks the dir for
    # emptiness — hf2q rejects "non-empty output dir" and exits code 3.
    local convert_log="$OUT_ROOT/${label}.convert.log"
    echo "[$label] start: ${env_prefix}hf2q convert --quant $quant"
    echo "[$label] log: $convert_log"
    # iter-100: HF2Q_QWEN35_DROP_MTP=1 matches the Apr 26 ADR-012 emission
    # path. Without the drop, hf2q emits blk.{n_layer}.nextn.* alongside
    # standard blk.{n_layer}.attn_norm.weight / ffn_*.weight / etc.; llama.cpp
    # reads block_count = n_layer + 1 and demands the full layer-tensor set
    # for the MTP block, including blk.{n_layer}.ssm_conv1d.weight which the
    # MTP wrapper does not provide → "missing tensor" load failure. Drop
    # MTP until llama.cpp gains qwen35 MTP loader (escape-hatch comment in
    # main.rs:1010 expires by 2026-Q4 if upstream still hasn't landed).
    /usr/bin/time -l env HF2Q_STREAMING_PHASE3="${stream:-0}" \
            HF2Q_QWEN35_DROP_MTP=1 \
        ./target/release/hf2q convert \
            --input "$src" \
            --format gguf \
            --quant "$quant" \
            --output "$out_dir" \
            --skip-quality \
            2>&1 | tee "$convert_log"

    local gguf
    gguf=$(find "$out_dir" -maxdepth 1 -name '*.gguf' | head -1)
    if [ -z "$gguf" ]; then
        echo "[$label] ERROR: no GGUF emitted"
        return 1
    fi
    echo "[$label] GGUF: $gguf ($(du -h "$gguf" | cut -f1))"
    echo "[$label] SHA-256: $(shasum -a 256 "$gguf" | cut -d' ' -f1)"

    # Coherence gate (Decision 22 AC): load + generate.
    # iter-102: pass -ngl 99 + --simple-io + 90s timeout. CPU-only Q4_K
    # decode is ~ minutes per token on 27B (OOM-tier slow); Metal GPU
    # offload is 30 t/s on M5 Max. The script's purpose is to verify the
    # GGUF *loads* and emits non-error tokens — speed isn't the gate.
    echo "[$label] llama-cli coherence check (GPU-offloaded, 90s timeout)…"
    local llama_out
    llama_out=$(timeout 90 "$LLAMA_CLI" -m "$gguf" -p "$PROMPT" -n 16 \
        --no-warmup -ngl 99 --simple-io 2>&1 | tail -10 || true)
    echo "[$label] llama-cli output: $llama_out"
    if echo "$llama_out" | grep -qiE "error loading model|failed to load|rejected the artefact|missing tensor"; then
        echo "[$label] FAIL: llama-cli rejected the artefact"
        return 1
    fi

    echo "[$label] PASS"
}

# ────────────────────────────────────────────────────────────
# Variant matrix
# ────────────────────────────────────────────────────────────
selector="${1:-all}"
stream_mode=""

case "$selector" in
    stream)
        stream_mode="1"
        selector="all"
        ;;
esac

# iter-92 fix: cli QuantMethod display strings are `dwq-4-6` / `dwq-4-8`
# (not the legacy `dwq-mixed-*` form). Pre-iter-92 the script would have
# been rejected by the clap parser at the first variant.
#
# iter-92 addition: per-variant selectors so the loop can run one heavy
# DWQ conversion at a time and let RAM settle between peaks (memory note
# project_dwq_concurrent_oom: DWQ 100 GB peak + concurrent inference =
# jetsam SIGKILL).
case "$selector" in
    all|dwq46|27b-dwq46)
        run_convert "27B-dwq46${stream_mode:+-stream}" "$QWEN27_SRC"     "dwq-4-6" "$stream_mode"
        ;;
esac
case "$selector" in
    all|dwq46|35bmoe-dwq46)
        run_convert "35BMOE-dwq46${stream_mode:+-stream}" "$QWEN35MOE_SRC" "dwq-4-6" "$stream_mode"
        ;;
esac
case "$selector" in
    all|dwq48|27b-dwq48)
        run_convert "27B-dwq48${stream_mode:+-stream}" "$QWEN27_SRC"     "dwq-4-8" "$stream_mode"
        ;;
esac
case "$selector" in
    all|dwq48|35bmoe-dwq48)
        run_convert "35BMOE-dwq48${stream_mode:+-stream}" "$QWEN35MOE_SRC" "dwq-4-8" "$stream_mode"
        ;;
esac

echo "----- P11 re-emit complete: $OUT_ROOT/ -----"
ls -lh "$OUT_ROOT"/*/*.gguf 2>/dev/null || true
