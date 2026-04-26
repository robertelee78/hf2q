#!/usr/bin/env bash
# ADR-005 Phase 2c iter-120 (W49) — F32 vs BF16 ViT attention A/B harness.
#
# Drives the same /v1/chat/completions request through hf2q serve twice:
#   - Run A: HF2Q_VIT_F32_ATTENTION unset (BF16 production path).
#   - Run B: HF2Q_VIT_F32_ATTENTION=1   (F32 path on the new mlx-native 0.4.7
#                                        dense_matmul_f32_f32_tensor primitive).
#
# Both runs use the same image (four_dots_in_corners_128x128.png) + prompt
# ("Describe this image in 5 words.") + T=0 + max_tokens=16 so the only
# variable is the K-side cast precision in vit_attention_scores_gpu.
#
# Output: writes the assistant message text (full content) for each run to
# /tmp/w49_bf16.log and /tmp/w49_f32.log respectively, plus the raw curl
# JSON to .json siblings for traceability.
#
# Sequential by design (one ~30 GB process at a time per OOM-prevention
# directive).
set -euo pipefail

HF2Q_BIN="/opt/hf2q/target/release/hf2q"
MODEL="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
MMPROJ="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf"
IMAGE="/opt/hf2q/tests/fixtures/vision/four_dots_in_corners_128x128.png"
PROMPT="Describe this image in 5 words."
PORT="18181"

run_one() {
    local label="$1"
    local env_setting="$2"
    local logf="/tmp/w49_${label}.log"
    local jsonf="/tmp/w49_${label}.json"
    local serve_log="/tmp/w49_${label}.serve.log"

    echo "[w49] === Run ${label} (env: ${env_setting:-unset}) ==="

    # Spawn server.
    if [ -n "$env_setting" ]; then
        eval "$env_setting" "$HF2Q_BIN" serve --model "$MODEL" --mmproj "$MMPROJ" --port "$PORT" --host 127.0.0.1 \
            > "$serve_log" 2>&1 &
    else
        env -u HF2Q_VIT_F32_ATTENTION "$HF2Q_BIN" serve --model "$MODEL" --mmproj "$MMPROJ" --port "$PORT" --host 127.0.0.1 \
            > "$serve_log" 2>&1 &
    fi
    local pid=$!

    # Wait for /readyz, ~600s ceiling.
    local ready=0
    for _ in $(seq 1 1200); do
        sleep 0.5
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/readyz" 2>/dev/null || echo "")
        if [ "$code" = "200" ]; then ready=1; break; fi
    done
    if [ "$ready" -ne 1 ]; then
        echo "[w49] ${label}: /readyz never returned 200; killing" >&2
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi

    # Resolve canonical chat model id from /v1/models. With --mmproj, the
    # response lists the mmproj entry first (id ends in '-mmproj'); the
    # chat-capable model is the one whose id does NOT end with '-mmproj'.
    local model_id
    model_id=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c \
        'import json,sys; ms=json.load(sys.stdin)["data"]; print(next(m["id"] for m in ms if not m["id"].endswith("-mmproj")))')

    # Build chat body with base64 image.
    local b64
    b64=$(base64 < "$IMAGE" | tr -d '\n')
    local body
    body=$(python3 - <<PY
import json
print(json.dumps({
    "model": "$model_id",
    "messages": [{"role":"user","content":[
        {"type":"text","text":"$PROMPT"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,$b64"}},
    ]}],
    "temperature": 0.0,
    "max_tokens": 16,
}))
PY
)

    # POST chat.
    curl -s -H 'Content-Type: application/json' \
        -d "$body" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" > "$jsonf"

    # Extract assistant message content.
    python3 -c \
        "import json; d=json.load(open('$jsonf')); print(d['choices'][0]['message']['content'])" \
        > "$logf" 2>/dev/null \
        || cp "$jsonf" "$logf"

    echo "[w49] ${label} text:"
    cat "$logf"
    echo ""

    # Tear down.
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    # Wait for full process clean.
    for _ in $(seq 1 60); do
        sleep 0.5
        if ! pgrep -f 'hf2q.*serve' >/dev/null 2>&1; then break; fi
    done
}

# --- Run A: BF16 (production default) ---
run_one "bf16" ""

# Confirm host quiet between runs.
sleep 2
if [ "$(pgrep -f 'hf2q.*serve' 2>/dev/null | wc -l | tr -d ' ')" != "0" ]; then
    echo "[w49] WARN: stray hf2q serve process detected between runs" >&2
    pkill -TERM -f 'hf2q.*serve' 2>/dev/null || true
    sleep 5
fi

# --- Run B: F32 attention ---
run_one "f32" "HF2Q_VIT_F32_ATTENTION=1"

echo "[w49] === SUMMARY ==="
echo "Run A (BF16): $(cat /tmp/w49_bf16.log)"
echo "Run B (F32 ): $(cat /tmp/w49_f32.log)"
