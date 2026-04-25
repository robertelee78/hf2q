#!/usr/bin/env bash
# ADR-005 Phase 2b iter-83 benchmark methodology — locked.
#
# Compares hf2q `/v1/embeddings` warm steady-state latency against
# llama-embedding's internal `prompt_eval` time on the same input
# ("hello world", 4 tokens, --pooling mean) using the same
# `nomic-embed-text-v1.5-f16.gguf`.
#
# Why these two engines this way:
#   - llama-embedding is run COLD (model load + forward + free) — its
#     internal stderr emits `prompt_eval = X ms / N tokens` which is
#     the GPU-forward-only cost AFTER load. We extract that line.
#   - hf2q is run as a long-lived server (model loaded once at boot,
#     then 20 warm `/v1/embeddings` requests via Python urllib with a
#     persistent connection). The first request fills the kernel
#     pipeline cache; we report the steady-state mean over requests
#     11-20 to exclude that warmup.
#
# Why curl was the wrong tool (iter-82 vs iter-83): each `curl`
# invocation is a fresh OS process + DNS + TCP connect + JSON encode
# adding ~150-180 ms of overhead per request that has nothing to do
# with the server. Iter-82 mistakenly attributed that to hf2q. Python
# urllib over a persistent session removes the curl noise floor.
#
# Usage:
#   bash scripts/bench_embedding.sh
#
# Env overrides:
#   HF2Q_BIN     binary path (default: ./target/release/hf2q)
#   PORT         hf2q server port (default: 39501)
#   NOMIC_GGUF   model path (default: /opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf)
#
# Exits 0 on success (prints comparison table), 1 on setup failure.

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-./target/release/hf2q}"
PORT="${PORT:-39501}"
NOMIC_GGUF="${NOMIC_GGUF:-/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf}"
LLAMA_EMBEDDING="${LLAMA_EMBEDDING:-/opt/homebrew/bin/llama-embedding}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[bench] hf2q binary not found at $HF2Q_BIN. Build first: cargo build --release --bin hf2q" >&2
    exit 1
fi
if [[ ! -f "$NOMIC_GGUF" ]]; then
    echo "[bench] Nomic GGUF not at $NOMIC_GGUF" >&2
    exit 1
fi
if [[ ! -x "$LLAMA_EMBEDDING" ]]; then
    echo "[bench] llama-embedding not at $LLAMA_EMBEDDING. Install via: brew install llama.cpp" >&2
    exit 1
fi

LOG_DIR="$(mktemp -d -t hf2q-bench-XXXXXX)"
HF2Q_LOG="$LOG_DIR/hf2q.log"

cleanup() {
    if [[ -n "${HF2Q_PID:-}" ]] && kill -0 "$HF2Q_PID" 2>/dev/null; then
        kill -TERM "$HF2Q_PID" 2>/dev/null || true
        wait "$HF2Q_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR"
}
trap cleanup EXIT INT TERM

echo "[bench] === llama-embedding cold runs (5×) ==="
LLAMA_LOG="$LOG_DIR/llama.log"
: > "$LLAMA_LOG"
for i in 1 2 3 4 5; do
    "$LLAMA_EMBEDDING" -m "$NOMIC_GGUF" -p "hello world" --pooling mean 2>"$LLAMA_LOG.$i" > /dev/null
    cat "$LLAMA_LOG.$i" >> "$LLAMA_LOG"
    PE_MS=$(grep 'prompt eval time' "$LLAMA_LOG.$i" | sed -E 's/.*=[[:space:]]*([0-9.]+)[[:space:]]*ms.*/\1/')
    TOTAL_MS=$(grep 'total time' "$LLAMA_LOG.$i" | sed -E 's/.*=[[:space:]]*([0-9.]+)[[:space:]]*ms.*/\1/')
    echo "  run $i: prompt_eval=${PE_MS}ms total=${TOTAL_MS}ms"
done
LLAMA_PE_MEAN=$(grep 'prompt eval time' "$LLAMA_LOG" | sed -E 's/.*=[[:space:]]*([0-9.]+)[[:space:]]*ms.*/\1/' | python3 -c "
import sys
vs = [float(line.strip()) for line in sys.stdin if line.strip()]
print(f'{sum(vs)/len(vs):.2f}')
")
echo "[bench]   prompt_eval mean: ${LLAMA_PE_MEAN}ms (forward-only, post model-load)"
echo

echo "[bench] === hf2q warm via persistent HTTP session (20 reqs) ==="
"$HF2Q_BIN" serve \
    --embedding-model "$NOMIC_GGUF" \
    --port "$PORT" \
    --host 127.0.0.1 \
    > "$HF2Q_LOG" 2>&1 &
HF2Q_PID=$!

# Wait for /health.
DEADLINE=$(( $(date +%s) + 30 ))
while true; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "[bench] FAIL: hf2q didn't come up in 30s" >&2
        cat "$HF2Q_LOG" >&2
        exit 1
    fi
    if curl -sSf -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 0.2
done

MODEL_ID="$(basename "$NOMIC_GGUF" .gguf)"

# Use python urllib with a persistent connection for clean
# per-request timing — see the comment block at the top for why
# curl-per-request is unsuitable for this measurement.
HF2Q_STEADY_STATE=$(python3 -c "
import urllib.request, json, time
import sys

PORT = $PORT
MODEL_ID = '$MODEL_ID'

url = f'http://127.0.0.1:{PORT}/v1/embeddings'
body = json.dumps({
    'model': MODEL_ID,
    'input': 'hello world',
    'encoding_format': 'float',
}).encode()

times = []
for i in range(20):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url, data=body, headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as r:
        _ = r.read()
    times.append((time.perf_counter() - t0) * 1000)

# Steady state = requests 11-20 (skip first 10 to amortize warmup).
ss = times[10:]
ss_mean = sum(ss) / len(ss)
ss_min = min(ss)
ss_max = max(ss)
ss_median = sorted(ss)[len(ss) // 2]

print(f'  per-req ms (20 reqs): {[f\"{t:.2f}\" for t in times]}', file=sys.stderr)
print(f'  steady-state (reqs 11-20): mean={ss_mean:.2f} min={ss_min:.2f} median={ss_median:.2f} max={ss_max:.2f}',
      file=sys.stderr)
print(ss_mean)
")
echo

echo "[bench] === comparison table ==="
HF2Q_FMT=$(python3 -c "print(f'{${HF2Q_STEADY_STATE}:.2f}')")
RATIO=$(python3 -c "print(f'{${HF2Q_STEADY_STATE} / ${LLAMA_PE_MEAN}:.2f}')")
printf "  %-42s %s\n" "llama.cpp internal prompt_eval (mean)"   "${LLAMA_PE_MEAN} ms"
printf "  %-42s %s\n" "hf2q HTTP /v1/embeddings (warm mean)"   "${HF2Q_FMT} ms"
printf "  %-42s %s\n" "ratio (hf2q full-stack / llama bare-GPU)" "${RATIO}×"

# Don't fail on ratio — this script is for measurement, not gating.
# A future iter may flip to a hard gate (e.g. ratio <= 2.5×).
echo
echo "[bench] DONE"
