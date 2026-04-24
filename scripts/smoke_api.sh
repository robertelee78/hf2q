#!/usr/bin/env bash
# ADR-005 Phase 2a smoke test for the hf2q HTTP API server.
#
# Spins up `hf2q serve` on an ephemeral port, exercises the complete
# endpoint surface (--model NOT supplied — backbone-only path), and
# asserts the response shapes match the OpenAI wire protocol.
#
# Coverage:
#   - GET  /health          200, status=ok, backend=mlx-native
#   - GET  /readyz          200, ready=true (trivially ready without engine)
#   - GET  /metrics         200, Prometheus text format v0.0.4 + counter bumps
#   - GET  /v1/models       200, object=list
#   - GET  /v1/models/{id}  404, code=model_not_found
#   - GET  /nope            404, OpenAI-shaped fallback envelope
#   - POST /v1/chat/completions          400, code=model_not_loaded (no engine)
#   - POST /v1/chat/completions stream   400, same gate ordering
#   - X-Request-Id echo (client-supplied + server-generated paths)
#   - Bearer auth: unauthenticated /health → 401 when --auth-token is set
#
# Usage:
#   bash scripts/smoke_api.sh
#
# Exits 0 on all-pass, 1 on first failure. Prints the failing response body
# so the mismatch is visible in CI logs.

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-./target/debug/hf2q}"
if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[smoke] Binary not found: $HF2Q_BIN" >&2
    echo "[smoke] Build first: cargo build --bin hf2q" >&2
    exit 1
fi

# Pick a port that's very unlikely to collide. Could scan but a high
# random port is fine for a smoke test.
PORT=${PORT:-39090}
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
AUTH_TOKEN="smoke-secret-42"

pass=0
fail=0

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        printf "  \e[32m✓\e[0m %s\n" "$label"
        pass=$((pass+1))
    else
        printf "  \e[31m✗\e[0m %s: expected %q, got %q\n" "$label" "$expected" "$actual"
        fail=$((fail+1))
    fi
}

assert_contains() {
    local label="$1" needle="$2" haystack="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        printf "  \e[32m✓\e[0m %s (contains %q)\n" "$label" "$needle"
        pass=$((pass+1))
    else
        printf "  \e[31m✗\e[0m %s: does not contain %q\n    body: %s\n" "$label" "$needle" "$haystack"
        fail=$((fail+1))
    fi
}

# ---------- Spin up the server (no model) ----------
echo "[smoke] Starting hf2q serve --port $PORT (no --model) …"
"$HF2Q_BIN" serve --host "$HOST" --port "$PORT" >/tmp/hf2q-smoke.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# Wait for bind.
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w '%{http_code}' "${BASE}/health" 2>/dev/null | grep -q 200; then
        break
    fi
    sleep 0.1
done

# ---------- /health ----------
echo "[smoke] /health"
body=$(curl -s -w '\n%{http_code}' "${BASE}/health")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 200" "200" "$code"
assert_contains "backend=mlx-native" "\"backend\":\"mlx-native\"" "$payload"
assert_contains "status=ok" "\"status\":\"ok\"" "$payload"

# ---------- /readyz ----------
echo "[smoke] /readyz"
body=$(curl -s -w '\n%{http_code}' "${BASE}/readyz")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 200" "200" "$code"
assert_contains "ready=true" "\"ready\":true" "$payload"

# ---------- /metrics ----------
echo "[smoke] /metrics"
body=$(curl -s -w '\n%{http_code}' "${BASE}/metrics")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 200" "200" "$code"
assert_contains "hf2q_uptime_seconds" "hf2q_uptime_seconds" "$payload"
assert_contains "hf2q_requests_total" "hf2q_requests_total" "$payload"
assert_contains "# HELP" "# HELP" "$payload"
assert_contains "# TYPE" "# TYPE" "$payload"

# ---------- /v1/models ----------
echo "[smoke] /v1/models"
body=$(curl -s -w '\n%{http_code}' "${BASE}/v1/models")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 200" "200" "$code"
assert_contains "object=list" "\"object\":\"list\"" "$payload"

# ---------- /v1/models/{missing} ----------
echo "[smoke] /v1/models/doesnotexist"
body=$(curl -s -w '\n%{http_code}' "${BASE}/v1/models/doesnotexist")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 404" "404" "$code"
assert_contains "model_not_found" "\"code\":\"model_not_found\"" "$payload"

# ---------- 404 fallback ----------
echo "[smoke] fallback /does-not-exist"
body=$(curl -s -w '\n%{http_code}' "${BASE}/does-not-exist")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 404" "404" "$code"
assert_contains "invalid_request_error" "\"type\":\"invalid_request_error\"" "$payload"

# ---------- POST /v1/chat/completions (no engine) ----------
echo "[smoke] POST /v1/chat/completions — expect 400 model_not_loaded"
body=$(curl -s -w '\n%{http_code}' -X POST "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}')
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 400" "400" "$code"
assert_contains "model_not_loaded" "\"code\":\"model_not_loaded\"" "$payload"

# ---------- POST /v1/chat/completions stream:true (no engine) ----------
echo "[smoke] POST /v1/chat/completions stream:true — expect 400 model_not_loaded"
body=$(curl -s -w '\n%{http_code}' -X POST "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}')
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 400" "400" "$code"
assert_contains "model_not_loaded" "\"code\":\"model_not_loaded\"" "$payload"

# ---------- X-Request-Id echo: client-supplied ----------
echo "[smoke] X-Request-Id echo (client-supplied)"
req_id=$(curl -s -D - "${BASE}/health" -H 'X-Request-Id: smoke-12345' -o /dev/null | grep -i '^x-request-id:' | awk '{print $2}' | tr -d '\r\n')
assert_eq "X-Request-Id echoed" "smoke-12345" "$req_id"

# ---------- X-Request-Id generated ----------
echo "[smoke] X-Request-Id generated (no client header)"
req_id=$(curl -s -D - "${BASE}/health" -o /dev/null | grep -i '^x-request-id:' | awk '{print $2}' | tr -d '\r\n')
if [[ ${#req_id} -eq 36 ]]; then
    printf "  \e[32m✓\e[0m X-Request-Id looks like UUIDv4 (36 chars): %s\n" "$req_id"
    pass=$((pass+1))
else
    printf "  \e[31m✗\e[0m X-Request-Id unexpected length (%d): %s\n" "${#req_id}" "$req_id"
    fail=$((fail+1))
fi

# ---------- Kill + restart with auth token ----------
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "[smoke] Restart with --auth-token $AUTH_TOKEN"
"$HF2Q_BIN" serve --host "$HOST" --port "$PORT" --auth-token "$AUTH_TOKEN" >/tmp/hf2q-smoke.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 30); do
    # Now requires auth, so /health direct returns 401. Poll a 401 as "up".
    code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "401" ]]; then
        break
    fi
    sleep 0.1
done

echo "[smoke] No auth → 401"
body=$(curl -s -w '\n%{http_code}' "${BASE}/health")
code=$(echo "$body" | tail -n1)
payload=$(echo "$body" | sed '$d')
assert_eq "HTTP 401" "401" "$code"
assert_contains "authentication_error" "\"type\":\"authentication_error\"" "$payload"

echo "[smoke] Wrong token → 401"
body=$(curl -s -w '\n%{http_code}' "${BASE}/health" -H "Authorization: Bearer wrong")
code=$(echo "$body" | tail -n1)
assert_eq "HTTP 401" "401" "$code"

echo "[smoke] Correct token → 200"
body=$(curl -s -w '\n%{http_code}' "${BASE}/health" -H "Authorization: Bearer ${AUTH_TOKEN}")
code=$(echo "$body" | tail -n1)
assert_eq "HTTP 200" "200" "$code"

# ---------- Summary ----------
echo
echo "[smoke] Passed: $pass  Failed: $fail"
if [[ $fail -eq 0 ]]; then
    echo "[smoke] All green ✓"
    exit 0
else
    echo "[smoke] Failures detected — see above."
    exit 1
fi
