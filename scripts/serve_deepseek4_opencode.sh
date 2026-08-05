#!/usr/bin/env bash
# serve_deepseek4_opencode.sh — canonical hf2q serve launcher for the
# DeepSeek-V4-Flash-0731 Q2_K_S GGUF, tuned for OpenCode agentic coding.
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
#   PORT=8084 scripts/serve_deepseek4_opencode.sh   # override port
#   CONTEXT_LEN=262144 scripts/serve_deepseek4_opencode.sh
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-Q2_K_S.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8083}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 3; }
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not found: $HF2Q_BIN (cargo build --release)" >&2; exit 3; }
if ! [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]] || (( CONTEXT_LEN < 128 )); then
    echo "CONTEXT_LEN must be an integer of at least 128 (got: $CONTEXT_LEN)" >&2
    exit 3
fi

# One-model-at-a-time guard: the Q2_K_S artifact is ~92 GiB and a second
# inference process on a 128 GiB unified-memory host will exhaust headroom.
if pgrep -x hf2q >/dev/null 2>&1; then
    echo "another hf2q process is already running — refusing to start a second" >&2
    echo "(kill it first: pkill -x hf2q)" >&2
    exit 1
fi

exec env \
    HF2Q_DEEPSEEK_MAX_SEQ_LEN="$CONTEXT_LEN" \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    "$HF2Q_BIN" serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --overflow-policy reject \
        --scheduler fifo-serial
