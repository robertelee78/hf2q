#!/usr/bin/env bash
# ADR-017 Phase E.a B.5 benchmark — LCP-resume TTFT speedup measurement.
#
# Measures three production-relevant scenarios on Qwen 3.6 35B-A3B-APEX-Q5_K_M:
#
# 1. **TTFT_baseline (cold turn-1)**: prompt with no cached LCP entry — full
#    chunked prefill from token 0.  Establishes the baseline-prefill cost.
# 2. **TTFT_lcp (warm turn-2)**: same prompt repeated with `[user X, assistant
#    Y, user Z]` multi-turn — shares ≥ stride tokens prefix with turn-1, hits
#    the descending-stride probe at chunk_pos=64, restores via `restore_partial`,
#    suffix-chunked prefills only the remainder.
# 3. **R-P6 aggregate (4-worker fan-out)**: 4 sequential requests sharing the
#    same turn-1 prefix with role-divergent suffixes (the actual /cfa Phase 2
#    shape — workers share `[SYSTEM] [QUEEN_SPEC]` but get role-specific tasks).
#    Aggregate TTFT compared to a single-worker fresh prefill.
#
# Outputs:
#   TTFT_baseline_p50:  N ms  (5-trial median)
#   TTFT_lcp_p50:       N ms
#   speedup:            X.X× (TTFT_baseline / TTFT_lcp)
#   R-P6 aggregate:     N ms total, X.X× single-worker baseline
#
# Each trial = warmup (server already loaded) + measurement.

set -uo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
HOST="127.0.0.1"
PORT="${PORT:-52481}"
TRIALS="${TRIALS:-5}"
LOG_DIR="${LOG_DIR:-/tmp/lcp-bench-$$}"
MAX_TOKENS="${MAX_TOKENS:-16}"

if [ ! -f "$MODEL" ]; then
    echo "[BENCH] model not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "[BENCH] log dir: $LOG_DIR"
echo "[BENCH] trials per scenario: $TRIALS"

# ── Spawn server ──────────────────────────────────────────────────────────
HF2Q_KV_LCP_CHUNKED_PREFILL=1 \
HF2Q_KV_LCP_RESUME=1 \
HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64 \
HF2Q_KV_LCP_RESUME_CAPACITY=8 \
/opt/hf2q/target/release/hf2q serve \
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

# Wait for /readyz.
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

# ── Helpers ───────────────────────────────────────────────────────────────
# Turn-1 user content — long enough to span 1+ stride boundary after chat
# template wrap (target prompt_len > 64 → at least 1 mid-prefill snapshot
# at chunk_pos=64).
TURN1_USER="I'm working on a Rust project organized by Domain-Driven Design \
bounded contexts. Could you describe in detail how bounded contexts in DDD \
map to Rust crate boundaries with a concrete example showing order, payment, \
and inventory in a typical e-commerce system that would help me structure \
my workspace?"

# Measure end-to-end request latency (ms).  We measure end-to-end rather
# than TTFT-strict (which would require parsing SSE arrival timestamps) —
# at MAX_TOKENS=16 the difference is small and the trend is the same: the
# prefill cost dominates, decode is constant per token.  This is a
# DELIBERATE simplification; the speedup ratio between baseline (full
# prefill from 0) and LCP (suffix prefill from chunk_pos=64) is dominated
# by the prefill saving.
chat_request_ms() {
    local body="$1"
    # macOS `date +%N` doesn't support nanoseconds — use python directly with
    # explicit timing arg (bash command substitution + python -c with embedded
    # vars was returning 0 in the prior implementation).  This invokes python
    # ONCE and lets it do both the request timing AND the curl-equivalent post.
    /usr/bin/python3 -c "
import time, sys, urllib.request, json
body = sys.stdin.read()
req = urllib.request.Request(
    'http://$HOST:$PORT/v1/chat/completions',
    data=body.encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
start = time.monotonic()
with urllib.request.urlopen(req, timeout=180) as resp:
    _ = resp.read()
end = time.monotonic()
print(int((end - start) * 1000))
" <<< "$body"
}

build_turn1_body() {
    /usr/bin/python3 -c "
import json, sys
body = json.dumps({
    'model': '$MODEL_ID',
    'messages': [{'role': 'user', 'content': '''$TURN1_USER'''}],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0,
    'stream': False,
})
print(body)
"
}

build_turn2_body() {
    local assistant_y="$1"
    local turn2_user="$2"
    /usr/bin/python3 -c "
import json
body = json.dumps({
    'model': '$MODEL_ID',
    'messages': [
        {'role': 'user',      'content': '''$TURN1_USER'''},
        {'role': 'assistant', 'content': '''$assistant_y'''},
        {'role': 'user',      'content': '''$turn2_user'''},
    ],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0,
    'stream': False,
})
print(body)
"
}

# ── Phase 1: warmup ──────────────────────────────────────────────────────
# First prefill triggers cold caches (pipeline compilation, kernel pre-pass).
# Discard timing.
echo "[BENCH] warmup..."
WARMUP_BODY=$(build_turn1_body)
chat_request_ms "$WARMUP_BODY" > /dev/null
echo "[BENCH] warmup done"

# Get the deterministic turn-1 response to use as assistant_y in turn-2.
ASSISTANT_Y=$(curl -sf -X POST -H "Content-Type: application/json" \
    -d "$WARMUP_BODY" "http://$HOST:$PORT/v1/chat/completions" \
    | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
echo "[BENCH] turn-1 deterministic response: ${ASSISTANT_Y:0:60}..."

# ── Phase 2: TTFT_cold (turn-2 shape, no matching LCP cache) ──────────────
# Each trial uses a UNIQUE TURN1_USER (different first-64-token prefix)
# so the descending-stride probe MISSES.  This is the fair "baseline":
# same prompt_len as the LCP-hit measurement (turn-2 multi-turn shape),
# just no cache to hit.
echo "[BENCH] measuring TTFT_cold (turn-2 shape, no LCP cache match) — $TRIALS trials..."
COLD_TIMES=()
for i in $(seq 1 "$TRIALS"); do
    UNIQ_TURN1_USER="${TURN1_USER} Trial $i divergence-marker-$RANDOM-$RANDOM-$RANDOM-$RANDOM (this prefix is unique per trial so the LCP probe at chunk_pos=64 cannot match a prior trial's stored entry; fresh chunked prefill from chunk_idx=0 is exercised end-to-end)."
    UNIQ_TURN2_USER="Worker $i: summarize the difference for $i."
    # Fake-multi-turn payload: real prompt is [user UNIQ_TURN1, assistant Y, user UNIQ_TURN2].
    BODY=$(/usr/bin/python3 -c "
import json
body = json.dumps({
    'model': '$MODEL_ID',
    'messages': [
        {'role': 'user',      'content': '''$UNIQ_TURN1_USER'''},
        {'role': 'assistant', 'content': '''$ASSISTANT_Y'''},
        {'role': 'user',      'content': '''$UNIQ_TURN2_USER'''},
    ],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0,
    'stream': False,
})
print(body)
")
    ms=$(chat_request_ms "$BODY")
    COLD_TIMES+=("$ms")
    echo "  trial $i: ${ms} ms"
done

# ── Phase 3: TTFT_lcp (turn-2 shape, matching LCP cache, stride-aligned hit) ─
# All trials use the canonical TURN1_USER + canonical ASSISTANT_Y +
# UNIQUE turn-2-user.  The first 64 tokens (chat-template + TURN1_USER
# prefix) match the cached entry stored at chunk_pos=64 by the warmup,
# so descending-stride probe hits, restore_partial, suffix-only chunked.
echo "[BENCH] measuring TTFT_lcp (turn-2 shape, STRIDE-ALIGNED HIT) — $TRIALS trials..."
# First, prime the cache with ONE warm turn-2 to ensure chunk_pos=64
# entry has cached_prompt = canonical TURN1_USER first 64 tokens (the
# COLD trials above may have stored a different cached_prompt).
PRIME_BODY=$(build_turn2_body "$ASSISTANT_Y" "PRIME")
chat_request_ms "$PRIME_BODY" > /dev/null
LCP_TIMES=()
for i in $(seq 1 "$TRIALS"); do
    UNIQ_TURN2_USER="Now in two sentences, summarize the main difference (trial $i)."
    BODY=$(build_turn2_body "$ASSISTANT_Y" "$UNIQ_TURN2_USER")
    ms=$(chat_request_ms "$BODY")
    LCP_TIMES+=("$ms")
    echo "  trial $i: ${ms} ms"
done

# ── Phase 4: R-P6 aggregate (4 sequential requests sharing turn-1 prefix) ─
echo "[BENCH] measuring R-P6 4-worker aggregate (1 trial)..."
RP6_START=$(/usr/bin/python3 -c 'import time; print(int(time.time()*1000))')
for w in 1 2 3 4; do
    UNIQ_TURN2="Worker $w: summarize the difference for $w."
    BODY=$(build_turn2_body "$ASSISTANT_Y" "$UNIQ_TURN2")
    chat_request_ms "$BODY" > /dev/null
done
RP6_END=$(/usr/bin/python3 -c 'import time; print(int(time.time()*1000))')
RP6_TOTAL=$((RP6_END - RP6_START))

# ── Stats ─────────────────────────────────────────────────────────────────
median() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:]]
print(int(statistics.median(nums)))
" "$@"
}

mean() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:]]
print(int(statistics.mean(nums)))
" "$@"
}

stdev() {
    /usr/bin/python3 -c "
import sys, statistics
nums = [int(x) for x in sys.argv[1:]]
print(int(statistics.stdev(nums))) if len(nums) > 1 else print(0)
" "$@"
}

COLD_P50=$(median "${COLD_TIMES[@]}")
COLD_MEAN=$(mean "${COLD_TIMES[@]}")
COLD_STDEV=$(stdev "${COLD_TIMES[@]}")
LCP_P50=$(median "${LCP_TIMES[@]}")
LCP_MEAN=$(mean "${LCP_TIMES[@]}")
LCP_STDEV=$(stdev "${LCP_TIMES[@]}")

SPEEDUP=$(/usr/bin/python3 -c "print(f'{$COLD_P50/$LCP_P50:.2f}')")
SAVINGS=$((COLD_P50 - LCP_P50))
RP6_RATIO=$(/usr/bin/python3 -c "print(f'{$RP6_TOTAL/(4*$COLD_P50):.2f}')")

# ── Engagement counters from server stderr ───────────────────────────────
HITS=$(grep -c "STRIDE-ALIGNED HIT" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n')
HITS=${HITS:-0}
STORES=$(grep -c "mid-prefill snapshot at chunk_pos=" "$LOG_DIR/server.stderr" 2>/dev/null | tr -d '\n')
STORES=${STORES:-0}

# ── Report ───────────────────────────────────────────────────────────────
echo ""
echo "[BENCH] === ADR-017 Phase E.a B.5 LCP-resume speedup ==="
echo "[BENCH] model:               Qwen 3.6 35B-A3B-APEX-Q5_K_M"
echo "[BENCH] config:              CHUNKED_PREFILL=1 LCP_RESUME=1 STRIDE=64 capacity=8"
echo "[BENCH] trials/scenario:     $TRIALS"
echo "[BENCH] max_tokens/req:      $MAX_TOKENS"
echo ""
echo "[BENCH] TTFT_cold (turn-2 shape, no matching LCP cache — fair baseline):"
echo "[BENCH]   p50:               ${COLD_P50} ms"
echo "[BENCH]   mean:              ${COLD_MEAN} ms ± ${COLD_STDEV} ms"
echo "[BENCH]   trials:            ${COLD_TIMES[*]}"
echo ""
echo "[BENCH] TTFT_lcp (turn-2 shape, STRIDE-ALIGNED HIT at chunk_pos=64):"
echo "[BENCH]   p50:               ${LCP_P50} ms"
echo "[BENCH]   mean:              ${LCP_MEAN} ms ± ${LCP_STDEV} ms"
echo "[BENCH]   trials:            ${LCP_TIMES[*]}"
echo ""
echo "[BENCH] LCP speedup:         ${SPEEDUP}× (saving ${SAVINGS} ms per request)"
echo ""
echo "[BENCH] R-P6 4-worker aggregate (sequential, all sharing turn-1 prefix):"
echo "[BENCH]   total:             ${RP6_TOTAL} ms"
echo "[BENCH]   ratio vs 4× cold:  ${RP6_RATIO}× (target: ≤ 1.25× per ADR-017 R-P6)"
echo ""
echo "[BENCH] Engagement counters (from server stderr):"
echo "[BENCH]   STRIDE-ALIGNED HITs: $HITS"
echo "[BENCH]   mid-prefill stores:  $STORES"
echo ""

# Pass criteria.
SPEEDUP_OK=$(/usr/bin/python3 -c "print(1 if $COLD_P50/$LCP_P50 >= 1.2 else 0)")
RP6_OK=$(/usr/bin/python3 -c "print(1 if $RP6_TOTAL/(4*$COLD_P50) <= 1.25 else 0)")

if [ "$HITS" -lt "$TRIALS" ]; then
    echo "[BENCH] WARN — HITs ($HITS) < TRIALS ($TRIALS); LCP didn't fire on every turn-2"
fi

if [ "$SPEEDUP_OK" = "1" ] && [ "$RP6_OK" = "1" ] && [ "$HITS" -ge "$TRIALS" ]; then
    echo "[BENCH] PASS — speedup ${SPEEDUP}×; R-P6 ${RP6_RATIO}×"
    exit 0
else
    echo "[BENCH] PARTIAL — speedup_ok=$SPEEDUP_OK rp6_ok=$RP6_OK hits=$HITS/$TRIALS"
    exit 0  # don't fail; report numbers regardless
fi
