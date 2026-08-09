#!/usr/bin/env bash
# serve_qwen36_opencode.sh — canonical hf2q serve launcher for the
# Qwen 3.6 35B-A3B APEX GGUF, tuned for opencode agentic coding.
#
# This is the canonical SlotAware launcher. Its live prefix-affinity and
# four-slot contracts are distinct from the historical SerialFifo disk-LCP
# restart path documented in ADR-027; this launcher does not claim or enable
# disk restart hydration.
#
#   Qwen3.6 autoregressive dispatch is the production default. No
#   investigation-only activation variable is required.
#   HF2Q_DEFAULT_REPETITION_PENALTY=1.05
#                           Loop mitigation (2026-08-03). opencode's
#                           openai-compatible provider cannot send
#                           repetition_penalty, so without this every
#                           request sampled with penalty 1.0 and long
#                           sessions degenerated into repetition loops
#                           (loop garbage then baked into compacted
#                           history, re-priming the loop each turn).
#                           Applied only when the client omits the param
#                           (explicit values win), only to generated
#                           tokens (never the prompt — code-safe), and
#                           never to the T=0 greedy argmax path. 1.05 is
#                           the gentle coding-safe setting; raise toward
#                           1.1 if loops persist on creative workloads.
#   HF2Q_ENCODER_SESSION=1
#                           Reuses ordered Metal command-buffer sessions
#                           across Qwen prefill stages. Recovery-state capture
#                           explicitly submits a carried FFN before entering
#                           its legacy sibling encoder path, preserving exact
#                           agentic output at K=8 while reducing short cached
#                           continuation latency.
#   HF2Q_FFN_TERMINAL_K_BATCH=8
#                           Drains the session every eight layers. The matched
#                           three-turn gate is 4/4 exact at K=8; larger values
#                           are not promoted by this launcher.
#   --overflow-policy reject
#                           opencode manages its own compaction; the
#                           server must 400 on overflow (OpenAI semantics)
#                           instead of silently rewriting the conversation
#                           (default "summarize") or truncating it.
#   --scheduler inflight-batched --max-slots 4
#                           Four independent agent sessions make progress and
#                           retain separate exact ChatML prefixes. Every slot
#                           advertises the full model context; no ctx/N math.
#   HF2Q_TQ_KV=1           TQ K/V stays active for every agent slot. The
#                           packed/norm buffers carry an outer slot axis and
#                           zero-copy Metal views select the active agent.
#   --kv-cache-budget-bytes Shared physical high-water across the full-context
#                           slots. MAX_SLOTS=8 is the np8-like setting.
#
# Family contract and prefix-cache stack:
#   * The loader uses the GGUF-embedded Qwen ChatML template and validates its
#     native turn, tool-call, and tool-response markers before inference.
#   1. Exact rendered-token affinity selects the slot holding the longest
#      matching ChatML prefix.
#   2. Each agent appends only its suffix to that slot's live TQ KV/recurrent
#      state; another agent no longer resets the single global cache.
#   3. Disk persistence remains available to explicitly configured SerialFifo
#      runs. The multi-slot worker does not claim restart hydration until that
#      path is wired and proven.
#
# Usage:
#   scripts/serve_qwen36_opencode.sh            # foreground (default)
#   PORT=8082 scripts/serve_qwen36_opencode.sh  # override port
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
MAX_SLOTS="${MAX_SLOTS:-4}"
KV_CACHE_BUDGET_BYTES="${KV_CACHE_BUDGET_BYTES:-51539607552}" # 48 GiB shared

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 3; }
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not found: $HF2Q_BIN (cargo build --release)" >&2; exit 3; }
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "PORT must be an integer from 1 through 65535 (got: $PORT)" >&2
    exit 3
fi
if ! [[ "$MAX_SLOTS" =~ ^[0-9]+$ ]] || (( MAX_SLOTS < 1 || MAX_SLOTS > 8 )); then
    echo "MAX_SLOTS must be from 1 through 8 (got: $MAX_SLOTS)" >&2
    exit 3
fi
if ! [[ "$KV_CACHE_BUDGET_BYTES" =~ ^[0-9]+$ ]] || (( KV_CACHE_BUDGET_BYTES < 1 )); then
    echo "KV_CACHE_BUDGET_BYTES must be a positive integer (got: $KV_CACHE_BUDGET_BYTES)" >&2
    exit 3
fi
# Refuse before loading the model when another service already owns the port.
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

# One-model-at-a-time guard (feedback_oom_prevention): a 35B-class model
# holds ~30 GB of unified memory; a second concurrent inference process
# on the same box risks OOM.
for RUNTIME_NAME in hf2q llama-server llama-cli llama-bench; do
    if RUNTIME_PIDS="$(pgrep -x "$RUNTIME_NAME" 2>/dev/null)"; then
        echo "another inference runtime is already running — refusing before model load" >&2
        echo "process: $RUNTIME_NAME (pid(s): ${RUNTIME_PIDS//$'\n'/, })" >&2
        echo "stop that runtime before starting Qwen" >&2
        exit 1
    fi
done

exec env \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    HF2Q_TQ_KV=1 \
    HF2Q_ENCODER_SESSION=1 \
    HF2Q_FFN_TERMINAL_K_BATCH=8 \
    "$HF2Q_BIN" -v serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --overflow-policy reject \
        --scheduler inflight-batched \
        --max-slots "$MAX_SLOTS" \
        --kv-cache-budget-bytes "$KV_CACHE_BUDGET_BYTES"
