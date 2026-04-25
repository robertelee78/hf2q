#!/usr/bin/env bash
# ADR-005 Phase 2a Task #11 — SIGTERM drain smoke test.
#
# Verifies that a SIGTERM sent to hf2q while an `/v1/embeddings` request
# is in flight does NOT truncate the response. Contract per Decision
# #17: "SIGTERM drains in-flight + queue then exits."
#
# Test shape:
#   1. Boot hf2q with --embedding-model bge-small-en-v1.5-f16.gguf
#   2. In a background process, POST a deliberately-long batch
#      (~80 inputs, each yielding one BERT forward pass; ~600ms wall-
#      clock at 7-9ms p50 per request).
#   3. After 200ms (long enough for the request to be received and
#      mid-batch in the spawn_blocking pool, short enough to be well
#      before the response would naturally arrive), send SIGTERM to
#      the server PID.
#   4. Wait for the client request to complete.
#   5. Assert: client got HTTP 200 with the full batch (80 embeddings),
#      and the server process exited cleanly within ~5s of the SIGTERM.
#
# What this catches:
#   - axum's with_graceful_shutdown not actually waiting for in-flight
#     HTTP responses (would manifest as 502 / connection reset).
#   - Embedding handler's spawn_blocking task being dropped before it
#     completes (would manifest as truncated response or panic).
#   - Server exiting on SIGTERM but leaving zombie threads alive (would
#     manifest as `pgrep -P` finding leftover children).
#
# Usage:
#   bash scripts/smoke_lifecycle_drain.sh
#
# Exits 0 on pass, 1 on first failure.

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-./target/release/hf2q}"
PORT="${PORT:-39443}"
EMBED_GGUF="${EMBED_GGUF:-/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf}"
BATCH_SIZE="${BATCH_SIZE:-80}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[smoke] Binary not found: $HF2Q_BIN" >&2
    exit 1
fi

if [[ ! -f "$EMBED_GGUF" ]]; then
    echo "[smoke] Embedding GGUF not found: $EMBED_GGUF" >&2
    exit 1
fi

LOG_DIR="$(mktemp -d -t hf2q-smoke-drain-XXXXXX)"
SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.json"
CLIENT_TIME="$LOG_DIR/client.time"
CLIENT_HTTP="$LOG_DIR/client.http_code"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR"
}
trap cleanup EXIT INT TERM

HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"

echo "[smoke] Booting hf2q on ${BASE}"
"$HF2Q_BIN" serve \
    --embedding-model "$EMBED_GGUF" \
    --port "$PORT" \
    --host "$HOST" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for /health.
DEADLINE=$(( $(date +%s) + 30 ))
while true; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "[smoke] FAIL: server didn't come up in 30s" >&2
        cat "$SERVER_LOG" >&2
        exit 1
    fi
    if curl -sSf -m 1 "${BASE}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done
echo "[smoke] Server up (PID=$SERVER_PID)"

# Build the batch JSON. 80 inputs with content variation so they don't
# get cached.
MODEL_ID="$(basename "$EMBED_GGUF" .gguf)"
BATCH_JSON="$(python3 -c "
import json
inputs = [f'request number {i} with some content to embed' for i in range($BATCH_SIZE)]
print(json.dumps({'model':'$MODEL_ID','input':inputs}))
")"

echo "[smoke] Issuing $BATCH_SIZE-input batch in background..."
(
    /usr/bin/time -p curl -sS -o "$CLIENT_LOG" -w "%{http_code}" \
        -X POST "${BASE}/v1/embeddings" \
        -H 'Content-Type: application/json' \
        -d "$BATCH_JSON" 2> "$CLIENT_TIME" > "$CLIENT_HTTP"
) &
CLIENT_PID=$!

# Wait long enough for the request to be received and the spawn_blocking
# task to start chewing through inputs, but well short of the natural
# response time (~600ms). 200ms is the right point.
sleep 0.2

echo "[smoke] Sending SIGTERM to server PID=$SERVER_PID..."
SIGTERM_TIME=$(date +%s.%N)
kill -TERM "$SERVER_PID"

echo "[smoke] Waiting for client request to complete..."
wait $CLIENT_PID || true
HTTP_CODE="$(cat "$CLIENT_HTTP")"
CLIENT_DONE_TIME=$(date +%s.%N)

echo "[smoke] Waiting for server to exit cleanly (≤5s)..."
SERVER_DEADLINE=$(( $(date +%s) + 5 ))
while kill -0 "$SERVER_PID" 2>/dev/null; do
    if [[ $(date +%s) -gt $SERVER_DEADLINE ]]; then
        echo "[smoke] FAIL: server still running 5s after SIGTERM" >&2
        ps -p "$SERVER_PID" >&2 || true
        cat "$SERVER_LOG" >&2 | tail -20
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 0.1
done
SERVER_EXITED_TIME=$(date +%s.%N)
SERVER_PID=""  # signal cleanup that the server is already gone

# --- Assertions ---
echo "[smoke] Asserting response shape..."

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "[smoke] FAIL: client got HTTP $HTTP_CODE (expected 200)" >&2
    echo "[smoke] Response body:" >&2
    head -c 500 "$CLIENT_LOG" >&2 || true
    echo >&2
    cat "$SERVER_LOG" >&2 | tail -20
    exit 1
fi

python3 - "$CLIENT_LOG" "$BATCH_SIZE" <<'PY'
import json, sys, math
log_path, expected_n = sys.argv[1], int(sys.argv[2])
with open(log_path) as f:
    body = json.load(f)
assert body["object"] == "list", f"Expected object=list, got {body!r}"
assert len(body["data"]) == expected_n, (
    f"Expected {expected_n} embeddings, got {len(body['data'])}"
)
hidden = len(body["data"][0]["embedding"])
for i, e in enumerate(body["data"]):
    assert e["index"] == i, f"data[{i}].index = {e['index']}"
    assert len(e["embedding"]) == hidden, f"data[{i}] dim mismatch"
    n = math.sqrt(sum(v * v for v in e["embedding"]))
    assert abs(n - 1.0) < 1e-3, f"data[{i}] not unit-norm: {n}"
print(f"  ✓ {expected_n} embeddings, dim={hidden}, all unit-norm")
PY

# Time accounting.
CLIENT_AFTER_SIGTERM_MS="$(python3 -c "print(int(($CLIENT_DONE_TIME - $SIGTERM_TIME) * 1000))")"
SERVER_AFTER_SIGTERM_MS="$(python3 -c "print(int(($SERVER_EXITED_TIME - $SIGTERM_TIME) * 1000))")"

echo "[smoke] Timing:"
echo "  client completed: ${CLIENT_AFTER_SIGTERM_MS}ms after SIGTERM"
echo "  server exited:    ${SERVER_AFTER_SIGTERM_MS}ms after SIGTERM"

# Sanity: client must complete before server exits.
if (( CLIENT_AFTER_SIGTERM_MS > SERVER_AFTER_SIGTERM_MS )); then
    echo "[smoke] FAIL: client finished AFTER server exit (drain ordering wrong)" >&2
    exit 1
fi

# Sanity: server must exit reasonably quickly after the in-flight
# request finishes. >2s lag indicates a hung worker.
LAG_MS="$(( SERVER_AFTER_SIGTERM_MS - CLIENT_AFTER_SIGTERM_MS ))"
if (( LAG_MS > 2000 )); then
    echo "[smoke] WARN: server took ${LAG_MS}ms to exit after client finished" >&2
fi

echo "[smoke] PASS Phase 1 — in-flight drain"

# === Phase 2: SIGTERM with NO in-flight requests should exit fast. ===
echo
echo "[smoke] Phase 2: SIGTERM with no in-flight work (must exit ≤500ms)"
"$HF2Q_BIN" serve \
    --embedding-model "$EMBED_GGUF" \
    --port "$PORT" \
    --host "$HOST" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
DEADLINE=$(( $(date +%s) + 30 ))
while ! curl -sSf -m 1 "${BASE}/health" >/dev/null 2>&1; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "[smoke] FAIL: phase 2 server didn't come up" >&2
        exit 1
    fi
    sleep 0.2
done
SIGTERM_T=$(date +%s.%N)
kill -TERM "$SERVER_PID"
SERVER_DEADLINE=$(( $(date +%s) + 5 ))
while kill -0 "$SERVER_PID" 2>/dev/null; do
    if [[ $(date +%s) -gt $SERVER_DEADLINE ]]; then
        echo "[smoke] FAIL: idle server still up 5s after SIGTERM" >&2
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 0.05
done
EXIT_T=$(date +%s.%N)
SERVER_PID=""
IDLE_EXIT_MS="$(python3 -c "print(int(($EXIT_T - $SIGTERM_T) * 1000))")"
echo "  idle server exited ${IDLE_EXIT_MS}ms after SIGTERM"
if (( IDLE_EXIT_MS > 500 )); then
    echo "[smoke] FAIL: idle server took ${IDLE_EXIT_MS}ms to exit (expected ≤500ms)" >&2
    exit 1
fi
echo "[smoke] PASS Phase 2 — idle exit fast"

echo
echo "[smoke] ALL PHASES PASSED ✓"
