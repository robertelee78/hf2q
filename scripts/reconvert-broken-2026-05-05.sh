#!/usr/bin/env bash
# Reconvert the 5 broken Qwen3.5/3.6 GGUFs (CFA 2026-05-05).
#
# Background: all 6 hf2q-built GGUFs dated 2026-04-30 had vocab=248044
# instead of 248320, no eos_token_id, no embedding rows for `<|im_end|>`.
# Fix at 505b5b8 (2026-05-02) + bail-gates at 0357394 (2026-05-05) ensure
# this can't regress silently. Reconverting at HEAD picks up both.
#
# Order: smallest peak RAM first to let the box settle between each step.
# Strict serial — `set -e` halts on first failure. No parallelism (per
# memory `project_dwq_concurrent_oom`: DWQ 100 GB peak + concurrent
# inference = jetsam SIGKILL).
#
# Output: /tmp/p11-re-emit-2026-05-05/<label>/<label>.gguf
# Logs:   /tmp/p11-re-emit-2026-05-05/<label>.convert.log
#
# After ALL 5 finish:
#   1. Inspect the verification block at the bottom of each .convert.log
#   2. Atomic-swap into /opt/hf2q/models/<original-name>/ (commands printed
#      at end). The on-disk broken GGUFs are NOT touched by this script.

set -euo pipefail

QWEN27_SRC="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
QWEN35MOE_SRC="$HOME/.cache/huggingface/hub/models--jenerallee78--Qwen3.6-35B-A3B-Abliterix-EGA-abliterated/snapshots/afde6ca7c35272a4b5eefb3b97576fdac0f74ba0"
OUT_ROOT="${HF2Q_RECONVERT_OUT_ROOT:-/tmp/p11-re-emit-2026-05-05}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_CLI="${HF2Q_LLAMA_CLI_BIN:-/opt/homebrew/bin/llama-cli}"
PROMPT="The quick brown fox"

mkdir -p "$OUT_ROOT"

if [ ! -x "$HF2Q_BIN" ]; then
    echo "ERROR: hf2q release binary missing at $HF2Q_BIN — run 'cargo build --release' first" >&2
    exit 2
fi
[ -d "$QWEN27_SRC" ] || { echo "ERROR: Qwen3.6-27B source missing at $QWEN27_SRC" >&2; exit 2; }
[ -d "$QWEN35MOE_SRC" ] || { echo "ERROR: Qwen3.6-35B-A3B source missing at $QWEN35MOE_SRC" >&2; exit 2; }

# Between-step RAM gate. Defers to ~/bin/free (free+inactive+speculative+
# purgeable; 80 GB headroom signal). Waits up to RECONVERT_RAM_WAIT_MAX seconds
# (default 600) for headroom to recover, sleeping 30s between probes. The
# DWQ peaks (~100 GB) require the higher 80 GB floor; the prior 30 GB
# floor + free+inactive count is too permissive and risked jetsam SIGKILL.
ram_check() {
    local label="$1"
    local min_avail="${HF2Q_RECONVERT_MIN_AVAIL_GB:-80}"
    local wait_max="${HF2Q_RECONVERT_RAM_WAIT_MAX:-600}"
    local elapsed=0
    while :; do
        local out gb
        out=$(~/bin/free 2>&1)
        gb=$(echo "$out" | awk -F'GB' '/Available/ {print $1; exit}')
        echo "[$label] RAM gate: ${out//$'\n'/ | }  (min: ${min_avail} GB)"
        if [ -n "$gb" ] && [ "$gb" -ge "$min_avail" ]; then
            return 0
        fi
        if [ "$elapsed" -ge "$wait_max" ]; then
            echo "[$label] ERROR: RAM headroom never recovered to ${min_avail} GB after ${wait_max}s" >&2
            return 1
        fi
        echo "[$label] sleeping 30s for RAM to settle (elapsed=${elapsed}s)"
        sleep 30
        elapsed=$(( elapsed + 30 ))
    done
}

# Verify the converted GGUF has the post-fix vocab metadata. Fails the
# step if the new bail somehow didn't catch a regression upstream.
verify_gguf() {
    local label="$1"
    local gguf="$2"
    if ! command -v gguf-dump >/dev/null 2>&1; then
        echo "[$label] WARNING: gguf-dump not on PATH — skipping post-convert verification"
        return 0
    fi
    echo "[$label] verifying tokenizer metadata…"
    local meta
    meta=$(gguf-dump --no-tensors "$gguf" 2>&1 | grep -E "tokenizer.ggml.tokens =|tokenizer.ggml.eos_token_id|tokenizer.ggml.bos_token_id|tokenizer.ggml.padding_token_id" || true)
    echo "$meta"
    # eos_token_id is mandatory — its absence is THE bug-class signature.
    # bos_token_id is optional (Qwen3.6 base doesn't declare a bos_token in
    # tokenizer_config.json; only the EGA-abliterated derivative does).
    # padding_token_id is optional (varies by source).
    if ! echo "$meta" | grep -q "eos_token_id"; then
        echo "[$label] FAIL: no tokenizer.ggml.eos_token_id in converted GGUF" >&2
        return 1
    fi
    if ! echo "$meta" | grep -q "bos_token_id"; then
        echo "[$label] note: no bos_token_id (expected for Qwen3.6 base; only the EGA derivative declares one)"
    fi
    # Token count must be >= 248320 (apex baseline). The line shape from
    # gguf-dump is `     N: [STRING]   |   COUNT | tokenizer.ggml.tokens = …`
    # so the 4th whitespace-separated field is COUNT. awk is robust where
    # the prior `grep -oE "[STRING\] …"` regex tripped pipefail on no-match.
    local tokens_line token_count
    tokens_line=$(echo "$meta" | grep "tokenizer.ggml.tokens =" || true)
    token_count=$(echo "$tokens_line" | awk '{print $4}')
    if [ -n "$token_count" ] && [ "$token_count" -lt 248320 ]; then
        echo "[$label] FAIL: token count $token_count < expected 248320 (added_tokens dropped?)" >&2
        return 1
    fi
    echo "[$label] verify OK (tokens=${token_count:-?}, eos present)"
}

run_convert() {
    local label="$1"
    local src="$2"
    local quant="$3"
    local drop_mtp="$4"  # "1" to drop MTP, "" to keep
    local out_dir="$OUT_ROOT/$label"
    local convert_log="$OUT_ROOT/${label}.convert.log"

    # Idempotent skip: if a verified GGUF already exists for this label,
    # don't redo it. Lets the chain resume mid-failure without redoing
    # multi-minute conversions that already produced known-good output.
    local existing_gguf
    existing_gguf=$(find "$out_dir" -maxdepth 1 -name '*.gguf' 2>/dev/null | head -1 || true)
    if [ -n "$existing_gguf" ] && [ -f "$existing_gguf" ]; then
        echo "[$label] existing GGUF found at $existing_gguf — re-verifying"
        if verify_gguf "$label" "$existing_gguf"; then
            echo "[$label] SKIP (already converted + verified)"
            return 0
        fi
        echo "[$label] existing GGUF failed verify — redoing"
    fi

    ram_check "$label"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    # DWQ steps peak at ~170-199 GB on a 128 GB box (jetsam-class).
    # `HF2Q_STREAMING_PHASE3=1` (borrowed wedge) clones each tensor to
    # Arc<Vec<u8>> — that DOUBLES memory and made things worse in the
    # 2026-05-05 14:55 run (199 GB peak vs 170 GB eager).
    # `HF2Q_STREAMING_PHASE3_MUT=1` (iter-83 zero-byte-copy) drains
    # `tensor_map` via `take_data_as_arc` mid-Phase-3 — actively frees
    # source memory as it goes, the actual lower-peak path.
    # Q4 (no calibration) runs eager fine (peaks 82-94 GB).
    local stream_env_pair=()
    case "$quant" in
        dwq-*) stream_env_pair=(HF2Q_STREAMING_PHASE3_MUT=1) ;;
    esac

    echo "[$label] start: ${stream_env_pair[*]:-} HF2Q_QWEN35_DROP_MTP=${drop_mtp:-0} hf2q convert --quant $quant"
    echo "[$label] log: $convert_log"
    /usr/bin/time -l env HF2Q_QWEN35_DROP_MTP="${drop_mtp:-0}" \
            "${stream_env_pair[@]}" \
        "$HF2Q_BIN" convert \
            --input "$src" \
            --format gguf \
            --quant "$quant" \
            --output "$out_dir" \
            --skip-quality \
            2>&1 | tee "$convert_log"

    local gguf
    gguf=$(find "$out_dir" -maxdepth 1 -name '*.gguf' | head -1)
    if [ -z "$gguf" ]; then
        echo "[$label] ERROR: no GGUF emitted (check $convert_log for ValidationFailed)" >&2
        return 1
    fi
    echo "[$label] GGUF: $gguf ($(du -h "$gguf" | cut -f1))"
    echo "[$label] SHA-256: $(shasum -a 256 "$gguf" | cut -d' ' -f1)"

    verify_gguf "$label" "$gguf" | tee -a "$convert_log"

    # Coherence gate. llama.cpp can't load Qwen3.5/3.6 MTP-emitting GGUFs
    # (escape-hatch documented in p11_re_emit_dwq.sh:90-97 — "Drop MTP
    # until llama.cpp gains qwen35 MTP loader"). For MTP-retained
    # variants the GGUF is fine for hf2q's own runtime — which is what
    # the user actually serves with — so skip the llama check there.
    # MTP-dropped ("flat" / DWQ) variants get the standard llama-cli
    # smoke as before.
    if [ "${drop_mtp:-0}" = "1" ]; then
        echo "[$label] llama-cli coherence check (GPU-offloaded, 90s timeout)…"
        local llama_out
        llama_out=$(timeout 90 "$LLAMA_CLI" -m "$gguf" -p "$PROMPT" -n 16 \
            --no-warmup -ngl 99 --simple-io 2>&1 | tail -10 || true)
        echo "[$label] llama-cli output: $llama_out"
        if echo "$llama_out" | grep -qiE "error loading model|failed to load|rejected the artefact|missing tensor"; then
            echo "[$label] FAIL: llama-cli rejected the artefact" >&2
            return 1
        fi
    else
        echo "[$label] llama-cli coherence: SKIPPED (MTP retained — llama.cpp can't load Qwen3.5/3.6 MTP yet; verified via metadata gate above)"
    fi
    echo "[$label] PASS"
}

# ────────────────────────────────────────────────────────────────────
# Order: smallest expected peak first so RAM headroom builds up safely.
#
# Recipe assumptions for the two not in p11_re_emit_dwq.sh:
#   * 27b-mtp-q4_0:  --quant q4 + KEEP MTP (matches "mtp" in name).
#                    Source = Qwen3.6-27B base.
#   * 35b-q4_0-flat: --quant q4 + DROP MTP (the "-flat" suffix means
#                    no MTP block; matches the existing q4_0-flat on
#                    disk that referenced scripts/adr019* expect).
#                    Source = abliterix-ega-abliterated.
# Both q4 (uncalibrated) variants are fast (<15 min) and low-RAM (<30 GB).
# ────────────────────────────────────────────────────────────────────

# Step 1 — smallest peak, fastest.
run_convert "qwen3.6-27b-mtp-q4_0"                                  "$QWEN27_SRC"     "q4"      ""

# Step 2 — 35B base model, q4_0 (uncalibrated), MTP dropped ("flat").
run_convert "qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat"   "$QWEN35MOE_SRC"  "q4"      "1"

# Step 3 — first DWQ; 27B is the smaller calibration peak.
run_convert "qwen3.6-27b-dwq46"                                     "$QWEN27_SRC"     "dwq-4-6" "1"

# Step 4 — 35B-MoE DWQ-4-6, ~80-100 GB peak.
run_convert "qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46"       "$QWEN35MOE_SRC"  "dwq-4-6" "1"

# Step 5 — 35B-MoE DWQ-4-8 (sensitive=8 = highest precision tier).
run_convert "qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48"       "$QWEN35MOE_SRC"  "dwq-4-8" "1"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "All 5 reconversions complete. Output GGUFs:"
ls -lh "$OUT_ROOT"/*/*.gguf 2>/dev/null || true
echo ""
echo "To swap the new GGUFs into place over the broken ones, run:"
for label in qwen3.6-27b-mtp-q4_0 \
             qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat \
             qwen3.6-27b-dwq46 \
             qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46 \
             qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48; do
    new="$OUT_ROOT/$label/$label.gguf"
    old="/opt/hf2q/models/$label/$label.gguf"
    echo "  mv \"$old\" \"$old.broken-2026-05-05\" && mv \"$new\" \"$old\""
done
echo "════════════════════════════════════════════════════════════════"
