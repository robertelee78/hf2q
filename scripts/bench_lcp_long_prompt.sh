#!/usr/bin/env bash
# ADR-017 Phase E.a perf-loop iter-3 bench — long-prompt + TTFT-strict + R-P6 arm.
#
# Mirrors scripts/bench_lcp_resume_speedup.sh but with two corrections to
# the iter-1 fixture's measurement floor:
#
# 1. **Long prompt (~5K tokens)** sourced from docs/ADR-017 head bytes.
#    Iter-1's 109-token fixture made decode dominate TTFT (~160 ms / 16 tok),
#    capping speedup at cold/decode ≈ 2.2×. A 5K-token shared prefix flips
#    the ratio: prefill ≈ 9 s vs decode ≈ 160 ms, so LCP can attack ~98%
#    of TTFT.
# 2. **TTFT-strict measurement** via SSE streaming (first content-delta arrival).
#    Gates on delta.content non-empty — skips role-only confirmation chunk that
#    hf2q emits before prefill completes (iter-2 TTFT bug, fixed in iter-3).
# 3. **R-P6 4-worker arm** (iter-3 addition): 4 sequential POSTs sharing the
#    long turn-1 prefix with role-divergent suffixes. Acceptance: ≤ 1.0× hard
#    (no regression vs 4× cold); ≤ 0.5× stretch (iter-2 prediction: ~0.27×).
#
# Per-trial instrumentation (iter-1 carryover): /metrics counter delta +
# server.stderr slice → per-trial K, cached_prompt_len, chunk_pos, restore_ms.

set -uo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
HOST="127.0.0.1"
PORT="${PORT:-52482}"
TRIALS="${TRIALS:-5}"
LOG_DIR="${LOG_DIR:-/tmp/lcp-long-bench-$$}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT_BYTES="${PROMPT_BYTES:-30000}"  # ~5K tokens at ~6 chars/token for ASCII Markdown
PROMPT_SOURCE="${PROMPT_SOURCE:-/opt/hf2q/docs/ADR-017-persistent-block-prefix-cache.md}"

if [ ! -f "$MODEL" ]; then
    echo "[BENCH] model not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$PROMPT_SOURCE" ]; then
    echo "[BENCH] prompt source not found: $PROMPT_SOURCE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "[BENCH] log dir: $LOG_DIR"
echo "[BENCH] trials per scenario: $TRIALS"
echo "[BENCH] prompt source: $PROMPT_SOURCE (${PROMPT_BYTES} bytes)"

# Build the long shared prompt (deterministic, ASCII-clean).
LONG_PROMPT_FILE="$LOG_DIR/long_prompt.txt"
head -c "$PROMPT_BYTES" "$PROMPT_SOURCE" > "$LONG_PROMPT_FILE"
PROMPT_CHARS=$(wc -c < "$LONG_PROMPT_FILE" | tr -d ' ')
PROMPT_WORDS=$(wc -w < "$LONG_PROMPT_FILE" | tr -d ' ')
echo "[BENCH] long prompt: ${PROMPT_CHARS} chars, ${PROMPT_WORDS} words"

# ── Spawn server ──────────────────────────────────────────────────────────
# caffeinate guard: prevents macOS App Nap from suspending the server during
# a long bench (laptop-lid-closed corrupted trial-5 in iter-2 at 131 s).
# Linux fall-through: CAFFEINATE is empty string (no-op prefix).
if [[ "$(uname -s)" == "Darwin" ]]; then
    CAFFEINATE="caffeinate -i"
else
    CAFFEINATE=""
fi
HF2Q_KV_LCP_CHUNKED_PREFILL=1 \
HF2Q_KV_LCP_RESUME=1 \
HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64 \
HF2Q_KV_LCP_RESUME_CAPACITY=8 \
$CAFFEINATE /opt/hf2q/target/release/hf2q serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    > "$LOG_DIR/server.stdout" 2> "$LOG_DIR/server.stderr" &
SERVER_PID=$!
echo "[BENCH] server pid=$SERVER_PID"

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 120); do
    if curl -sf "http://$HOST:$PORT/readyz" > /dev/null 2>&1; then
        echo "[BENCH] server ready after ${i}s"
        break
    fi
    sleep 1
done
if ! curl -sf "http://$HOST:$PORT/readyz" > /dev/null 2>&1; then
    echo "[BENCH] server failed to come up" >&2
    exit 1
fi

MODEL_ID=$(curl -s "http://$HOST:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[BENCH] model id: $MODEL_ID"

# ── Per-trial instrumentation helpers (iter-2 carryover) ─────────────────
trial_metrics_lookups() {
    curl -sf "http://$HOST:$PORT/metrics" 2>/dev/null \
        | awk '/^hf2q_kv_lcp_lookups_total / {print $2}'
}
trial_metrics_detected() {
    curl -sf "http://$HOST:$PORT/metrics" 2>/dev/null \
        | awk '/^hf2q_kv_lcp_detected_total / {print $2}'
}
trial_stderr_lines() {
    wc -l < "$LOG_DIR/server.stderr" 2>/dev/null | tr -d ' '
}
trial_parse_stderr_slice() {
    local from="$1" to="$2"
    local slice
    slice=$(sed -n "${from},${to}p" "$LOG_DIR/server.stderr" 2>/dev/null)
    local hit=0
    local k="" cached_len="" chunk_pos="" restore_ms="" prompt_len=""
    if printf '%s\n' "$slice" | grep -q "STRIDE-ALIGNED HIT"; then
        hit=1
        local hit_line
        hit_line=$(printf '%s\n' "$slice" | grep "STRIDE-ALIGNED HIT" | head -1)
        k=$(printf '%s' "$hit_line" | grep -oE 'k=[0-9]+' | head -1 | cut -d= -f2)
        cached_len=$(printf '%s' "$hit_line" | grep -oE 'cached_prompt_len=[0-9]+' | head -1 | cut -d= -f2)
        chunk_pos=$(printf '%s' "$hit_line" | grep -oE 'chunk_pos=[0-9]+' | head -1 | cut -d= -f2)
        restore_ms=$(printf '%s' "$hit_line" | grep -oE 'restore_ms=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
    fi
    # Extract observed prompt_len from the probe-enabled log line.
    local probe_line
    probe_line=$(printf '%s\n' "$slice" | grep -E "lcp probe\] enabled" | head -1)
    prompt_len=$(printf '%s' "$probe_line" | grep -oE 'prompt_len=[0-9]+' | head -1 | cut -d= -f2)
    echo "${hit},${k},${cached_len},${chunk_pos},${restore_ms},${prompt_len}"
}

# ── TTFT-strict streaming chat request ────────────────────────────────────
# Returns "<ttft_ms> <total_ms>" — TTFT = time-to-first-data-line in SSE
# stream (first content delta), total = full request wall.
chat_request_streaming_ms() {
    local body="$1"
    HOST="$HOST" PORT="$PORT" /usr/bin/python3 -c "
import os, time, sys, urllib.request, json
body = sys.stdin.read()
req = urllib.request.Request(
    f'http://{os.environ[\"HOST\"]}:{os.environ[\"PORT\"]}/v1/chat/completions',
    data=body.encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
start = time.monotonic()
ttft = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode('utf-8', errors='replace').rstrip()
        if not line.startswith('data: '):
            continue
        payload = line[6:]
        if payload == '[DONE]':
            continue
        # First content-delta arrival = TTFT-strict.
        try:
            j = json.loads(payload)
            if ttft is None:
                # Take any first SSE data line that isn't a role-only chunk.
                # Some servers emit role-first chunk before content; prefer
                # first content delta. Fall back to first chunk if needed.
                delta = j.get('choices', [{}])[0].get('delta', {})
                if delta.get('content') is not None and delta.get('content') != '':
                    ttft = time.monotonic() - start
        except Exception:
            pass
total = time.monotonic() - start
ttft_ms = int(ttft * 1000) if ttft is not None else -1
total_ms = int(total * 1000)
print(f'{ttft_ms} {total_ms}')
" <<< "$body"
}

build_turn1_body() {
    MODEL_ID="$MODEL_ID" CONTENT_FILE="$LONG_PROMPT_FILE" MAX_TOKENS="$MAX_TOKENS" \
    /usr/bin/python3 -c "
import json, os
with open(os.environ['CONTENT_FILE'], 'r') as f:
    content = f.read()
print(json.dumps({
    'model': os.environ['MODEL_ID'],
    'messages': [{'role': 'user', 'content': content}],
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': 0,
    'stream': True,
}))
"
}

build_turn2_body() {
    local assistant_y="$1"
    local turn2_user="$2"
    local turn1_prepend="${3:-}"
    MODEL_ID="$MODEL_ID" \
    CONTENT_FILE="$LONG_PROMPT_FILE" \
    TURN1_PREPEND="$turn1_prepend" \
    ASSISTANT_Y="$assistant_y" \
    TURN2_USER="$turn2_user" \
    MAX_TOKENS="$MAX_TOKENS" \
    /usr/bin/python3 -c "
import json, os
with open(os.environ['CONTENT_FILE'], 'r') as f:
    base = f.read()
turn1 = (os.environ.get('TURN1_PREPEND', '') + base) if os.environ.get('TURN1_PREPEND') else base
print(json.dumps({
    'model': os.environ['MODEL_ID'],
    'messages': [
        {'role': 'user',      'content': turn1},
        {'role': 'assistant', 'content': os.environ['ASSISTANT_Y']},
        {'role': 'user',      'content': os.environ['TURN2_USER']},
    ],
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': 0,
    'stream': True,
}))
"
}

# ── Phase 1: warmup ──────────────────────────────────────────────────────
echo "[BENCH] warmup (this may take 10-30s on a long prompt)..."
WARMUP_BODY=$(build_turn1_body)
chat_request_streaming_ms "$WARMUP_BODY" > /dev/null
echo "[BENCH] warmup done"

# Get the deterministic turn-1 response. Use non-streaming for clean parsing.
ASSISTANT_Y=$(MODEL_ID="$MODEL_ID" CONTENT_FILE="$LONG_PROMPT_FILE" MAX_TOKENS="$MAX_TOKENS" \
    /usr/bin/python3 -c "
import json, os, urllib.request
with open(os.environ['CONTENT_FILE'], 'r') as f:
    content = f.read()
body = json.dumps({
    'model': os.environ['MODEL_ID'],
    'messages': [{'role': 'user', 'content': content}],
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': 0,
    'stream': False,
}).encode('utf-8')
req = urllib.request.Request(
    f'http://${HOST}:${PORT}/v1/chat/completions',
    data=body, headers={'Content-Type': 'application/json'}, method='POST'
)
with urllib.request.urlopen(req, timeout=600) as resp:
    j = json.loads(resp.read())
print(j['choices'][0]['message']['content'])
")
echo "[BENCH] turn-1 deterministic response: ${ASSISTANT_Y:0:80}..."

# ── Phase 2: TTFT_cold (turn-2 shape, unique prepend per trial) ──────────
echo "[BENCH] measuring TTFT_cold (turn-2 shape, no LCP cache match) — $TRIALS trials..."
COLD_TTFT=()
COLD_TOTAL=()
COLD_LOOKUPS=()
COLD_DETECTED=()
COLD_HIT=()
COLD_K=()
COLD_CHUNK=()
COLD_RESTORE=()
COLD_PROMPT_LEN=()
for i in $(seq 1 "$TRIALS"); do
    PREPEND="V$i-$RANDOM TRIAL $i SEED. "
    UNIQ_TURN2_USER="Worker $i: in two sentences, summarize the most important takeaway."
    BODY=$(build_turn2_body "$ASSISTANT_Y" "$UNIQ_TURN2_USER" "$PREPEND")
    LOOKUPS_BEFORE=$(trial_metrics_lookups); LOOKUPS_BEFORE=${LOOKUPS_BEFORE:-0}
    DETECTED_BEFORE=$(trial_metrics_detected); DETECTED_BEFORE=${DETECTED_BEFORE:-0}
    STDERR_BEFORE=$(trial_stderr_lines); STDERR_BEFORE=${STDERR_BEFORE:-0}
    RESULT=$(chat_request_streaming_ms "$BODY")
    TTFT_MS=$(echo "$RESULT" | awk '{print $1}')
    TOTAL_MS=$(echo "$RESULT" | awk '{print $2}')
    LOOKUPS_AFTER=$(trial_metrics_lookups); LOOKUPS_AFTER=${LOOKUPS_AFTER:-0}
    DETECTED_AFTER=$(trial_metrics_detected); DETECTED_AFTER=${DETECTED_AFTER:-0}
    STDERR_AFTER=$(trial_stderr_lines); STDERR_AFTER=${STDERR_AFTER:-0}
    LOOKUPS_DELTA=$((LOOKUPS_AFTER - LOOKUPS_BEFORE))
    DETECTED_DELTA=$((DETECTED_AFTER - DETECTED_BEFORE))
    PARSED=$(trial_parse_stderr_slice "$((STDERR_BEFORE+1))" "$STDERR_AFTER")
    IFS=',' read -r T_HIT T_K T_CACHED T_CHUNK T_RESTORE T_PROMPT_LEN <<< "$PARSED"
    COLD_TTFT+=("$TTFT_MS")
    COLD_TOTAL+=("$TOTAL_MS")
    COLD_LOOKUPS+=("$LOOKUPS_DELTA")
    COLD_DETECTED+=("$DETECTED_DELTA")
    COLD_HIT+=("$T_HIT")
    COLD_K+=("${T_K:--}")
    COLD_CHUNK+=("${T_CHUNK:--}")
    COLD_RESTORE+=("${T_RESTORE:--}")
    COLD_PROMPT_LEN+=("${T_PROMPT_LEN:--}")
    echo "  trial $i: ttft=${TTFT_MS}ms total=${TOTAL_MS}ms (lookups+${LOOKUPS_DELTA} detected+${DETECTED_DELTA} hit=${T_HIT} k=${T_K:--} chunk=${T_CHUNK:--} restore_ms=${T_RESTORE:--} prompt_len=${T_PROMPT_LEN:--})"
done

# ── Phase 3: prime + TTFT_lcp (turn-2 shape, canonical prefix) ───────────
echo "[BENCH] priming canonical first-N tokens via one warm turn-2..."
PRIME_BODY=$(build_turn2_body "$ASSISTANT_Y" "PRIME" "")
chat_request_streaming_ms "$PRIME_BODY" > /dev/null

echo "[BENCH] measuring TTFT_lcp (turn-2 shape, STRIDE-ALIGNED HIT) — $TRIALS trials..."
LCP_TTFT=()
LCP_TOTAL=()
LCP_LOOKUPS=()
LCP_DETECTED=()
LCP_HIT=()
LCP_K=()
LCP_CHUNK=()
LCP_RESTORE=()
LCP_PROMPT_LEN=()
for i in $(seq 1 "$TRIALS"); do
    UNIQ_TURN2_USER="In two sentences, give me the most important takeaway (trial $i)."
    BODY=$(build_turn2_body "$ASSISTANT_Y" "$UNIQ_TURN2_USER" "")
    LOOKUPS_BEFORE=$(trial_metrics_lookups); LOOKUPS_BEFORE=${LOOKUPS_BEFORE:-0}
    DETECTED_BEFORE=$(trial_metrics_detected); DETECTED_BEFORE=${DETECTED_BEFORE:-0}
    STDERR_BEFORE=$(trial_stderr_lines); STDERR_BEFORE=${STDERR_BEFORE:-0}
    RESULT=$(chat_request_streaming_ms "$BODY")
    TTFT_MS=$(echo "$RESULT" | awk '{print $1}')
    TOTAL_MS=$(echo "$RESULT" | awk '{print $2}')
    LOOKUPS_AFTER=$(trial_metrics_lookups); LOOKUPS_AFTER=${LOOKUPS_AFTER:-0}
    DETECTED_AFTER=$(trial_metrics_detected); DETECTED_AFTER=${DETECTED_AFTER:-0}
    STDERR_AFTER=$(trial_stderr_lines); STDERR_AFTER=${STDERR_AFTER:-0}
    LOOKUPS_DELTA=$((LOOKUPS_AFTER - LOOKUPS_BEFORE))
    DETECTED_DELTA=$((DETECTED_AFTER - DETECTED_BEFORE))
    PARSED=$(trial_parse_stderr_slice "$((STDERR_BEFORE+1))" "$STDERR_AFTER")
    IFS=',' read -r T_HIT T_K T_CACHED T_CHUNK T_RESTORE T_PROMPT_LEN <<< "$PARSED"
    LCP_TTFT+=("$TTFT_MS")
    LCP_TOTAL+=("$TOTAL_MS")
    LCP_LOOKUPS+=("$LOOKUPS_DELTA")
    LCP_DETECTED+=("$DETECTED_DELTA")
    LCP_HIT+=("$T_HIT")
    LCP_K+=("${T_K:--}")
    LCP_CHUNK+=("${T_CHUNK:--}")
    LCP_RESTORE+=("${T_RESTORE:--}")
    LCP_PROMPT_LEN+=("${T_PROMPT_LEN:--}")
    echo "  trial $i: ttft=${TTFT_MS}ms total=${TOTAL_MS}ms (lookups+${LOOKUPS_DELTA} detected+${DETECTED_DELTA} hit=${T_HIT} k=${T_K:--} chunk=${T_CHUNK:--} restore_ms=${T_RESTORE:--} prompt_len=${T_PROMPT_LEN:--})"
done

# ── Phase 4: R-P6 aggregate (4 sequential requests sharing turn-1 long prefix) ─
# Modeled on bench_lcp_resume_speedup.sh:303-329 shape — single python timing
# process issues all 4 requests to avoid cross-subprocess time.monotonic()
# inconsistency. Workers share the long turn-1 prefix; each gets a unique
# role-divergent turn-2 suffix ("Worker N: summarize the difference for N.").
# Predicted from iter-2 data: 1×cold (~19 s) + 3×LCP (~0.4 s each) ≈ 20.2 s
# vs 4×cold = 76 s → R-P6 ratio ≈ 0.27× (well under both 1.0× hard and 0.5×
# stretch acceptance bands).
echo "[BENCH] measuring R-P6 4-worker aggregate (long-prompt, 1 trial)..."
RP6_BODIES_FILE=$(mktemp)
for w in 1 2 3 4; do
    UNIQ_TURN2_USER="Worker $w: summarize the difference for $w."
    build_turn2_body "$ASSISTANT_Y" "$UNIQ_TURN2_USER" "" >> "$RP6_BODIES_FILE"
    echo "" >> "$RP6_BODIES_FILE"
done
RP6_TOTAL=$(HOST="$HOST" PORT="$PORT" BODIES="$RP6_BODIES_FILE" /usr/bin/python3 -c "
import os, time, urllib.request
url = f'http://{os.environ[\"HOST\"]}:{os.environ[\"PORT\"]}/v1/chat/completions'
with open(os.environ['BODIES'], 'r') as f:
    bodies = [b for b in f.read().split('\n') if b.strip()]
start = time.monotonic()
for body in bodies:
    req = urllib.request.Request(
        url, data=body.encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        _ = resp.read()
print(int((time.monotonic() - start) * 1000))
")
rm -f "$RP6_BODIES_FILE"
echo "[BENCH] R-P6 4-worker total: ${RP6_TOTAL} ms"

# ── Stats helpers ─────────────────────────────────────────────────────────
median() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:] if x and x != '-1']
print(int(statistics.median(nums)) if nums else -1)
" "$@"
}
mean() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:] if x and x != '-1']
print(int(statistics.mean(nums)) if nums else -1)
" "$@"
}
stdev() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:] if x and x != '-1']
print(int(statistics.stdev(nums)) if len(nums) > 1 else 0)
" "$@"
}

COLD_TTFT_P50=$(median "${COLD_TTFT[@]}")
COLD_TTFT_MEAN=$(mean "${COLD_TTFT[@]}")
COLD_TTFT_STD=$(stdev "${COLD_TTFT[@]}")
LCP_TTFT_P50=$(median "${LCP_TTFT[@]}")
LCP_TTFT_MEAN=$(mean "${LCP_TTFT[@]}")
LCP_TTFT_STD=$(stdev "${LCP_TTFT[@]}")
COLD_TOTAL_P50=$(median "${COLD_TOTAL[@]}")
LCP_TOTAL_P50=$(median "${LCP_TOTAL[@]}")

if [ "$COLD_TTFT_P50" -gt 0 ] && [ "$LCP_TTFT_P50" -gt 0 ]; then
    TTFT_SPEEDUP=$(/usr/bin/python3 -c "print(f'{$COLD_TTFT_P50/$LCP_TTFT_P50:.2f}')")
else
    TTFT_SPEEDUP="N/A"
fi
if [ "$COLD_TOTAL_P50" -gt 0 ] && [ "$LCP_TOTAL_P50" -gt 0 ]; then
    TOTAL_SPEEDUP=$(/usr/bin/python3 -c "print(f'{$COLD_TOTAL_P50/$LCP_TOTAL_P50:.2f}')")
else
    TOTAL_SPEEDUP="N/A"
fi
if [ "$COLD_TOTAL_P50" -gt 0 ]; then
    RP6_RATIO=$(/usr/bin/python3 -c "print(f'{$RP6_TOTAL/(4*$COLD_TOTAL_P50):.2f}')")
else
    RP6_RATIO="N/A"
fi

# ── Report ───────────────────────────────────────────────────────────────
echo ""
echo "[BENCH] === ADR-017 Phase E.a perf-loop iter-3 long-prompt bench ==="
echo "[BENCH] model:               $(basename "$MODEL")"
echo "[BENCH] config:              CHUNKED_PREFILL=1 LCP_RESUME=1 STRIDE=64 capacity=8"
echo "[BENCH] trials/scenario:     $TRIALS"
echo "[BENCH] long-prompt source:  $PROMPT_SOURCE (${PROMPT_BYTES} bytes)"
echo "[BENCH] max_tokens/req:      $MAX_TOKENS"
echo ""
echo "[BENCH] TTFT-strict (first SSE content-delta arrival; role-only chunk skipped):"
echo "[BENCH]   COLD p50:          ${COLD_TTFT_P50} ms (mean ${COLD_TTFT_MEAN} ± ${COLD_TTFT_STD})"
echo "[BENCH]   LCP  p50:          ${LCP_TTFT_P50} ms (mean ${LCP_TTFT_MEAN} ± ${LCP_TTFT_STD})"
echo "[BENCH]   speedup:           ${TTFT_SPEEDUP}× (TTFT-strict)"
echo "[BENCH]   acceptance:        ≥ 5× hard line (long-prompt: prefill dominates TTFT)"
echo ""
echo "[BENCH] Total request wall (TTFT + decode of $MAX_TOKENS tokens):"
echo "[BENCH]   COLD p50:          ${COLD_TOTAL_P50} ms"
echo "[BENCH]   LCP  p50:          ${LCP_TOTAL_P50} ms"
echo "[BENCH]   speedup:           ${TOTAL_SPEEDUP}× (full request)"
echo ""
echo "[BENCH] R-P6 4-worker aggregate (sequential, sharing long turn-1 prefix):"
echo "[BENCH]   total:             ${RP6_TOTAL} ms"
echo "[BENCH]   ratio vs 4× cold:  ${RP6_RATIO}×"
echo "[BENCH]   acceptance:        ≤ 1.0× hard (no regression vs 4× cold)"
echo "[BENCH]   stretch:           ≤ 0.5× (iter-2 prediction ~0.27×: 1×cold + 3×LCP ≈ 20 s vs 76 s)"
echo ""
echo "[BENCH] Per-trial breakdown (loop iter-3):"
printf "[BENCH]   COLD %-3s | %-6s | %-6s | %-7s | %-8s | %-3s | %-5s | %-5s | %-10s | %s\n" \
    "i" "ttft" "total" "lookups" "detected" "hit" "k" "chunk" "restore_ms" "prompt_len"
for i in $(seq 0 $((${#COLD_TTFT[@]} - 1))); do
    printf "[BENCH]      %-3s | %-6s | %-6s | %-7s | %-8s | %-3s | %-5s | %-5s | %-10s | %s\n" \
        "$((i+1))" \
        "${COLD_TTFT[$i]}" \
        "${COLD_TOTAL[$i]}" \
        "${COLD_LOOKUPS[$i]}" \
        "${COLD_DETECTED[$i]}" \
        "${COLD_HIT[$i]}" \
        "${COLD_K[$i]}" \
        "${COLD_CHUNK[$i]}" \
        "${COLD_RESTORE[$i]}" \
        "${COLD_PROMPT_LEN[$i]}"
done
echo ""
printf "[BENCH]   LCP  %-3s | %-6s | %-6s | %-7s | %-8s | %-3s | %-5s | %-5s | %-10s | %s\n" \
    "i" "ttft" "total" "lookups" "detected" "hit" "k" "chunk" "restore_ms" "prompt_len"
for i in $(seq 0 $((${#LCP_TTFT[@]} - 1))); do
    printf "[BENCH]      %-3s | %-6s | %-6s | %-7s | %-8s | %-3s | %-5s | %-5s | %-10s | %s\n" \
        "$((i+1))" \
        "${LCP_TTFT[$i]}" \
        "${LCP_TOTAL[$i]}" \
        "${LCP_LOOKUPS[$i]}" \
        "${LCP_DETECTED[$i]}" \
        "${LCP_HIT[$i]}" \
        "${LCP_K[$i]}" \
        "${LCP_CHUNK[$i]}" \
        "${LCP_RESTORE[$i]}" \
        "${LCP_PROMPT_LEN[$i]}"
done
echo ""

# ── Acceptance block ──────────────────────────────────────────────────────
# TTFT speedup hard line: ≥ 5× on long-prompt (prefill dominates at ~5K tokens;
# decode floor is ~160 ms vs cold prefill ~9-19 s → structural ceiling ~60-120×;
# 5× is a deliberately conservative floor given the 57× measured in iter-2).
# R-P6 hard line: ≤ 1.0× of 4× cold (no regression vs running 4 cold requests).
# Numbers are computed from actual COLD/LCP arrays — never hard-coded.
TTFT_PASS=0
RP6_PASS=0
if [ "$TTFT_SPEEDUP" != "N/A" ]; then
    TTFT_PASS=$(/usr/bin/python3 -c "print(1 if float('$TTFT_SPEEDUP') >= 5.0 else 0)")
fi
if [ "$RP6_RATIO" != "N/A" ]; then
    RP6_PASS=$(/usr/bin/python3 -c "print(1 if float('$RP6_RATIO') <= 1.0 else 0)")
fi

HITS=$(grep -c "STRIDE-ALIGNED HIT" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n')
HITS=${HITS:-0}
if [ "$HITS" -lt "$TRIALS" ]; then
    echo "[BENCH] WARN — LCP HITs ($HITS) < TRIALS ($TRIALS); LCP did not fire on every turn-2"
fi

if [ "$TTFT_PASS" = "1" ] && [ "$RP6_PASS" = "1" ]; then
    echo "[BENCH] PASS — TTFT_speedup=${TTFT_SPEEDUP}× (≥ 5× hard); R-P6_ratio=${RP6_RATIO}× (≤ 1.0× hard)"
    exit 0
else
    echo "[BENCH] FAIL — TTFT_speedup=${TTFT_SPEEDUP}× (need ≥ 5×, pass=${TTFT_PASS}); R-P6_ratio=${RP6_RATIO}× (need ≤ 1.0×, pass=${RP6_PASS})"
    exit 1
fi
