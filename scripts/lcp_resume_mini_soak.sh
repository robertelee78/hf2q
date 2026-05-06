#!/usr/bin/env bash
# ADR-017 Phase E.a B.5 mini-soak — 5-minute LCP-resume regression check.
#
# Spawns ONE server with HF2Q_KV_LCP_CHUNKED_PREFILL=1 + HF2Q_KV_LCP_RESUME=1
# + HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64 + HF2Q_KV_LCP_RESUME_CAPACITY=4,
# then runs ~50 multi-turn chat sequences (turn-1 + turn-2) measuring:
#   * Sustained correctness — every turn-2 response decoded non-empty + non-error.
#   * Memory pressure — RSS before/after (operator should see no growth beyond
#     ~5 GB for the 4-entry registry + per-request kv_cache churn).
#   * Engagement rate — count of `STRIDE-ALIGNED HIT` log lines vs total turns.
#
# This is NOT the 24h soak — it's a session-length regression check that
# verifies the LCP-resume path doesn't leak or fail after sustained use.
# Operators run the actual 24h on /cfa workloads with default capacity=8.

set -uo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
HOST="127.0.0.1"
PORT="${PORT:-52471}"
TURNS="${TURNS:-50}"
LOG_DIR="${LOG_DIR:-/tmp/lcp-mini-soak-$$}"

if [ ! -f "$MODEL" ]; then
    echo "[MINI-SOAK] model not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "[MINI-SOAK] log dir: $LOG_DIR"
echo "[MINI-SOAK] turns: $TURNS"

# ── Spawn server ──────────────────────────────────────────────────────────
HF2Q_KV_LCP_CHUNKED_PREFILL=1 \
HF2Q_KV_LCP_RESUME=1 \
HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64 \
HF2Q_KV_LCP_RESUME_CAPACITY=4 \
/opt/hf2q/target/release/hf2q serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    > "$LOG_DIR/server.stdout" 2> "$LOG_DIR/server.stderr" &
SERVER_PID=$!
echo "[MINI-SOAK] server pid=$SERVER_PID"

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for /readyz.
for i in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/readyz" > /dev/null 2>&1; then
        echo "[MINI-SOAK] server ready after ${i}s"
        break
    fi
    sleep 1
done

# Get canonical model id.
MODEL_ID=$(curl -s "http://$HOST:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[MINI-SOAK] model id: $MODEL_ID"

# Initial memory.
RSS_START=$(ps -o rss= -p "$SERVER_PID" | tr -d ' ')
echo "[MINI-SOAK] RSS_start = ${RSS_START} KB"

# ── Run multi-turn chat sequences ─────────────────────────────────────────
TURN1_USER="I'm working on a Rust project organized by Domain-Driven Design bounded contexts. Could you describe in detail how bounded contexts in DDD map to Rust crate boundaries with a concrete example showing order, payment, and inventory in a typical e-commerce system that would help me structure my workspace?"

OK_COUNT=0
FAIL_COUNT=0
# `grep -c` exits 1 (no matches) with stdout="0" — capture as plain int.
HIT_COUNT_BEFORE=$(grep -c "STRIDE-ALIGNED HIT" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n' | sed 's/[^0-9]//g')
HIT_COUNT_BEFORE=${HIT_COUNT_BEFORE:-0}

for t in $(seq 1 "$TURNS"); do
    # turn-1 (non-streaming).
    REQ_T1=$(jq -nc \
      --arg model "$MODEL_ID" \
      --arg content "$TURN1_USER" \
      '{model: $model, messages: [{role: "user", content: $content}], max_tokens: 16, temperature: 0, stream: false}')
    RESP_T1=$(curl -sf -X POST -H "Content-Type: application/json" \
        -d "$REQ_T1" "http://$HOST:$PORT/v1/chat/completions" 2>/dev/null || echo "ERROR")
    if [[ "$RESP_T1" == "ERROR" ]] || ! echo "$RESP_T1" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        echo "[MINI-SOAK] turn $t: turn-1 FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    A_Y=$(echo "$RESP_T1" | jq -r '.choices[0].message.content')

    # turn-2 (streaming for variety).
    REQ_T2=$(jq -nc \
      --arg model "$MODEL_ID" \
      --arg user1 "$TURN1_USER" \
      --arg assistant_y "$A_Y" \
      --arg user2 "Now in two sentences, summarize the main difference." \
      '{model: $model, messages: [
        {role: "user", content: $user1},
        {role: "assistant", content: $assistant_y},
        {role: "user", content: $user2}
      ], max_tokens: 16, temperature: 0, stream: true}')
    RESP_T2=$(curl -sf -X POST -H "Content-Type: application/json" \
        -d "$REQ_T2" "http://$HOST:$PORT/v1/chat/completions" 2>/dev/null || echo "ERROR")
    if [[ "$RESP_T2" == "ERROR" ]]; then
        echo "[MINI-SOAK] turn $t: turn-2 FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    # Extract content from SSE stream.
    T2_BYTES=$(echo "$RESP_T2" | grep -E '"content":"' | wc -c)
    if [ "$T2_BYTES" -lt 10 ]; then
        echo "[MINI-SOAK] turn $t: turn-2 EMPTY"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    OK_COUNT=$((OK_COUNT + 1))
    if (( t % 10 == 0 )); then
        RSS_NOW=$(ps -o rss= -p "$SERVER_PID" | tr -d ' ')
        echo "[MINI-SOAK] turn $t/$TURNS ok=$OK_COUNT fail=$FAIL_COUNT RSS=${RSS_NOW} KB"
    fi
done

# ── Final stats ───────────────────────────────────────────────────────────
RSS_END=$(ps -o rss= -p "$SERVER_PID" | tr -d ' ')
HIT_COUNT_AFTER=$(grep -c "STRIDE-ALIGNED HIT" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n' | sed 's/[^0-9]//g')
HIT_COUNT_AFTER=${HIT_COUNT_AFTER:-0}
HIT_DELTA=$((HIT_COUNT_AFTER - HIT_COUNT_BEFORE))

ERROR_COUNT=$(grep -cE "panicked|ERROR|GenerationError|Generation failed" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n' | sed 's/[^0-9]//g')
ERROR_COUNT=${ERROR_COUNT:-0}

echo ""
echo "[MINI-SOAK] === FINAL ==="
echo "[MINI-SOAK] turns:       $TURNS"
echo "[MINI-SOAK] ok:          $OK_COUNT"
echo "[MINI-SOAK] failed:      $FAIL_COUNT"
echo "[MINI-SOAK] STRIDE hits: $HIT_DELTA"
echo "[MINI-SOAK] RSS_start:   ${RSS_START} KB"
echo "[MINI-SOAK] RSS_end:     ${RSS_END} KB"
echo "[MINI-SOAK] RSS_delta:   $((RSS_END - RSS_START)) KB"
echo "[MINI-SOAK] errors:      $ERROR_COUNT (panics/Generation-failed in stderr)"
echo ""

if [ "$FAIL_COUNT" -gt 0 ] || [ "$ERROR_COUNT" -gt 0 ]; then
    echo "[MINI-SOAK] FAIL"
    exit 1
fi

# Memory check: allow up to 25% growth for cache fill (4-entry registry × ~300 MB per snapshot).
RSS_DELTA=$((RSS_END - RSS_START))
RSS_LIMIT=$((RSS_START / 4))
if [ "$RSS_DELTA" -gt "$RSS_LIMIT" ]; then
    echo "[MINI-SOAK] WARN — RSS grew by ${RSS_DELTA} KB (>25% of start ${RSS_START} KB)"
fi

echo "[MINI-SOAK] PASS"
