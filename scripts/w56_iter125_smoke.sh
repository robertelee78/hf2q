#!/usr/bin/env bash
# ADR-005 Phase 2c iter-125 (W56) — smoke test for the preprocess scale-bias fix.
# Two cold-process T=0 runs against the four-dots fixture; verifies determinism
# and captures verbatim output to compare against peer truth.
set -euo pipefail

HF2Q_BIN="/opt/hf2q/target/release/hf2q"
MODEL="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
MMPROJ="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf"
IMAGE="/opt/hf2q/tests/fixtures/vision/four_dots_in_corners_128x128.png"
PROMPT="Describe this image in 5 words."
PORT="18181"

run_one() {
    local label="$1"
    local logf="/tmp/w56_${label}.log"
    local jsonf="/tmp/w56_${label}.json"
    local serve_log="/tmp/w56_${label}.serve.log"

    echo "[w56] === Run ${label} ==="

    "$HF2Q_BIN" serve --model "$MODEL" --mmproj "$MMPROJ" --port "$PORT" --host 127.0.0.1 \
        > "$serve_log" 2>&1 &
    local pid=$!

    local ready=0
    for _ in $(seq 1 1200); do
        sleep 0.5
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/readyz" 2>/dev/null || echo "")
        if [ "$code" = "200" ]; then ready=1; break; fi
    done
    if [ "$ready" -ne 1 ]; then
        echo "[w56] ${label}: /readyz never returned 200; killing" >&2
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi

    local model_id
    model_id=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c \
        'import json,sys; ms=json.load(sys.stdin)["data"]; print(next(m["id"] for m in ms if not m["id"].endswith("-mmproj")))')

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

    curl -s -H 'Content-Type: application/json' \
        -d "$body" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" > "$jsonf"

    python3 -c \
        "import json; d=json.load(open('$jsonf')); print(d['choices'][0]['message']['content'])" \
        > "$logf" 2>/dev/null \
        || cp "$jsonf" "$logf"

    echo "[w56] ${label} text:"
    cat "$logf"
    echo ""

    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    for _ in $(seq 1 60); do
        sleep 0.5
        if ! pgrep -f 'hf2q.*serve' >/dev/null 2>&1; then break; fi
    done
}

run_one "run1"
sleep 2
if [ "$(pgrep -f 'hf2q.*serve' 2>/dev/null | wc -l | tr -d ' ')" != "0" ]; then
    pkill -TERM -f 'hf2q.*serve' 2>/dev/null || true
    sleep 5
fi
run_one "run2"

echo "[w56] === SUMMARY ==="
echo "Run 1: $(cat /tmp/w56_run1.log)"
echo "Run 2: $(cat /tmp/w56_run2.log)"
