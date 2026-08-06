#!/usr/bin/env bash
# serve_deepseek4_opencode.sh — canonical hf2q serve launcher for the
# DeepSeek-V4-Flash-0731 agentic Q2/Q3/Q8 GGUF, tuned for OpenCode coding.
#
# The DeepSeek cache is native and in-memory:
#
#   HF2Q_DEEPSEEK_MAX_SEQ_LEN=131072
#                           Allocates the live cache plus one equally sized
#                           prompt-tail recovery checkpoint (~1.71 GiB total
#                           at 131K). Growing transcripts prefill only their
#                           suffix; reasoning canonicalization restores the
#                           checkpoint and replays the rewritten tail.
#   HF2Q_DEFAULT_REPETITION_PENALTY=1.05
#                           Gentle loop mitigation when a sampling client omits
#                           repetition_penalty. Explicit request values win;
#                           temperature-zero greedy decoding is unchanged.
#   --overflow-policy reject
#                           OpenCode manages conversation compaction, so the
#                           server reports context overflow instead of silently
#                           rewriting or truncating the transcript.
#   --scheduler fifo-serial
#                           Required by the current single-session DeepSeek
#                           prefix-cache worker. Concurrent requests queue;
#                           slot-aware serving is rejected explicitly.
#
# Unlike the Qwen launcher, this does not set HF2Q_KV_LCP_RESUME or configure
# --kv-persist. DeepSeek-V4 owns a family-specific live cache and recovery
# checkpoint; it currently does not serialize them across server restarts.
# No external converter or inference runtime is used.
#
# Usage:
#   scripts/serve_deepseek4_opencode.sh             # foreground (default)
#   PORT=8090 scripts/serve_deepseek4_opencode.sh   # override port
#   CONTEXT_LEN=262144 scripts/serve_deepseek4_opencode.sh
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-agentic-q2.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 3; }
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not found: $HF2Q_BIN (cargo build --release)" >&2; exit 3; }
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "PORT must be an integer from 1 through 65535 (got: $PORT)" >&2
    exit 3
fi
if ! [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]] || (( CONTEXT_LEN < 128 )); then
    echo "CONTEXT_LEN must be an integer of at least 128 (got: $CONTEXT_LEN)" >&2
    exit 3
fi

# Fail before loading the ~100 GiB model when another service owns the port.
# macOS ships lsof; nc is a portable fallback for leaner environments.
if command -v lsof >/dev/null 2>&1; then
    PORT_LISTENER="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$PORT_LISTENER" ]]; then
        echo "$HOST:$PORT is already in use — refusing before model load" >&2
        printf '%s\n' "$PORT_LISTENER" >&2
        echo "choose a free port, for example: PORT=8090 $0" >&2
        exit 2
    fi
elif command -v nc >/dev/null 2>&1 && nc -z "$HOST" "$PORT" >/dev/null 2>&1; then
    echo "$HOST:$PORT is already in use — refusing before model load" >&2
    echo "choose a free port, for example: PORT=8090 $0" >&2
    exit 2
fi

# One-model-at-a-time guard: the agentic artifact is ~100 GiB and a second
# inference process on a 128 GiB unified-memory host will exhaust headroom.
for RUNTIME_NAME in hf2q llama-server llama-cli llama-bench; do
    if RUNTIME_PIDS="$(pgrep -x "$RUNTIME_NAME" 2>/dev/null)"; then
        echo "another inference runtime is already running — refusing before model load" >&2
        echo "process: $RUNTIME_NAME (pid(s): ${RUNTIME_PIDS//$'\n'/, })" >&2
        echo "stop that runtime before starting DeepSeek-V4" >&2
        exit 1
    fi
done

exec env \
    HF2Q_DEEPSEEK_MAX_SEQ_LEN="$CONTEXT_LEN" \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    "$HF2Q_BIN" serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --overflow-policy reject \
        --scheduler fifo-serial
