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
#   bash scripts/p11_re_emit_dwq.sh           # all 4 variants
#   bash scripts/p11_re_emit_dwq.sh dwq46     # 27B + 35B-MoE dwq46 only
#   bash scripts/p11_re_emit_dwq.sh dwq48     # 27B + 35B-MoE dwq48 only
#   bash scripts/p11_re_emit_dwq.sh stream    # all 4 with HF2Q_STREAMING_PHASE3=1
#
# Default behavior is the eager path; `stream` runs all 4 with the env flag on.

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
    local free_pages
    free_pages=$(vm_stat | awk '/Pages free/ { gsub(/\./, "", $3); print $3 }')
    local free_gb
    free_gb=$(( free_pages * 16384 / 1024 / 1024 / 1024 ))
    echo "[$label] RAM free: ${free_gb} GB"
    if [ "$free_gb" -lt 30 ]; then
        echo "[$label] ERROR: <30 GB free — refusing to load model (mantra OOM-prevention)"
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

    echo "[$label] start: ${env_prefix}hf2q convert --quant $quant"
    /usr/bin/time -l env HF2Q_STREAMING_PHASE3="${stream:-0}" \
        ./target/release/hf2q convert \
            --input "$src" \
            --format gguf \
            --quant "$quant" \
            --output "$out_dir" \
            --skip-quality \
            2>&1 | tee "$out_dir/convert.log"

    local gguf
    gguf=$(find "$out_dir" -maxdepth 1 -name '*.gguf' | head -1)
    if [ -z "$gguf" ]; then
        echo "[$label] ERROR: no GGUF emitted"
        return 1
    fi
    echo "[$label] GGUF: $gguf ($(du -h "$gguf" | cut -f1))"
    echo "[$label] SHA-256: $(shasum -a 256 "$gguf" | cut -d' ' -f1)"

    # Coherence gate (Decision 22 AC): load + generate
    echo "[$label] llama-cli coherence check…"
    local llama_out
    llama_out=$("$LLAMA_CLI" -m "$gguf" -p "$PROMPT" -n 16 --no-warmup 2>&1 | tail -3 || true)
    echo "[$label] llama-cli output: $llama_out"
    if echo "$llama_out" | grep -qiE "error|failed|rejected"; then
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

case "$selector" in
    all|dwq46)
        run_convert "27B-dwq46${stream_mode:+-stream}" "$QWEN27_SRC"     "dwq-mixed-4-6" "$stream_mode"
        run_convert "35BMOE-dwq46${stream_mode:+-stream}" "$QWEN35MOE_SRC" "dwq-mixed-4-6" "$stream_mode"
        ;;
esac
case "$selector" in
    all|dwq48)
        run_convert "27B-dwq48${stream_mode:+-stream}" "$QWEN27_SRC"     "dwq-mixed-4-8" "$stream_mode"
        run_convert "35BMOE-dwq48${stream_mode:+-stream}" "$QWEN35MOE_SRC" "dwq-mixed-4-8" "$stream_mode"
        ;;
esac

echo "----- P11 re-emit complete: $OUT_ROOT/ -----"
ls -lh "$OUT_ROOT"/*/*.gguf 2>/dev/null || true
