#!/usr/bin/env bash
# serve_deepseek4_opencode.sh — canonical hf2q serve launcher for the
# DeepSeek-V4-Flash-0731 agentic Q2/Q3/Q8 GGUF, tuned for OpenCode coding.
#
# The DeepSeek cache is native and in-memory:
#
#   HF2Q_DEEPSEEK_MAX_SEQ_LEN=524288
#                           Advertises a 512K serving limit while initially
#                           allocating a 131K live cache (~877 MiB). Capacity
#                           grows in 131K steps only when the transcript needs
#                           it; the prompt-tail checkpoint stays ~17 MiB.
#                           Growing transcripts prefill only their
#                           suffix; reasoning canonicalization restores the
#                           checkpoint and replays the rewritten tail.
#   HF2Q_DEFAULT_REPETITION_PENALTY=1.05
#                           Gentle loop mitigation when a sampling client omits
#                           repetition_penalty. Explicit request values win;
#                           temperature-zero greedy decoding is unchanged.
#   HF2Q_DEEPSEEK_PREFILL_WINDOWS=adaptive
#                           Uses the measured 2,048-token transaction while
#                           the live cache is 131K, then 1,024 after capacity
#                           grows. Set PREFILL_WINDOWS explicitly only when
#                           benchmarking a measured alternative.
#   --overflow-policy reject
#                           OpenCode manages conversation compaction, so the
#                           server reports context overflow instead of silently
#                           rewriting or truncating the transcript.
#   --scheduler fifo-serial
#                           Required by the current single-session DeepSeek
#                           prefix-cache worker. Concurrent requests queue;
#                           slot-aware serving is rejected explicitly.
#   -v                      Enables request/cache/prefill/decode progress at
#                           info level. Direct `hf2q serve` remains quiet by
#                           default; this foreground launcher is observable.
#
# Unlike the Qwen launcher, this does not set HF2Q_KV_LCP_RESUME or configure
# --kv-persist. DeepSeek-V4 owns a family-specific live cache and recovery
# checkpoint; it currently does not serialize them across server restarts.
# No external converter or inference runtime is used.
#
# Usage:
#   scripts/serve_deepseek4_opencode.sh             # foreground (default)
#   PORT=8090 scripts/serve_deepseek4_opencode.sh   # override port
#   CONTEXT_LEN=131072 scripts/serve_deepseek4_opencode.sh  # lower-memory
#   CONTEXT_LEN=1048576 scripts/serve_deepseek4_opencode.sh # full trained window
#   CHECK_ONLY=1 scripts/serve_deepseek4_opencode.sh        # preflight only
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-agentic-q2.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONTEXT_LEN="${CONTEXT_LEN:-524288}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# A ~100 GiB resident model leaves little margin on a 128 GiB host. Refuse a
# load when prior inference has left the compressor/swap saturated or one
# competing process already consumes a material part of that margin. These
# limits are intentionally conservative; the explicit unsafe override exists
# for controlled diagnostics, not normal OpenCode serving.
MAX_SWAP_USED_GIB="${MAX_SWAP_USED_GIB:-8}"
MAX_COMPRESSOR_USED_GIB="${MAX_COMPRESSOR_USED_GIB:-8}"
MAX_OTHER_PROCESS_RSS_GIB="${MAX_OTHER_PROCESS_RSS_GIB:-8}"
UNSAFE_MEMORY_OVERRIDE="${HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE:-0}"
LARGE_MODEL_BYTES=$((80 * 1024 * 1024 * 1024))

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
for SETTING in CHECK_ONLY MAX_SWAP_USED_GIB MAX_COMPRESSOR_USED_GIB \
    MAX_OTHER_PROCESS_RSS_GIB UNSAFE_MEMORY_OVERRIDE; do
    VALUE=${!SETTING}
    if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
        echo "$SETTING must be a non-negative integer (got: $VALUE)" >&2
        exit 3
    fi
done
if (( CHECK_ONLY > 1 || UNSAFE_MEMORY_OVERRIDE > 1 )); then
    echo "CHECK_ONLY and HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE must be 0 or 1" >&2
    exit 3
fi
if [[ -n "${PREFILL_WINDOWS:-}" ]]; then
    if ! [[ "$PREFILL_WINDOWS" =~ ^[1-9][0-9]*$ ]]; then
        echo "PREFILL_WINDOWS must be a positive integer (got: $PREFILL_WINDOWS)" >&2
        exit 3
    fi
    export HF2Q_DEEPSEEK_PREFILL_WINDOWS="$PREFILL_WINDOWS"
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

MODEL_BYTES=$(stat -f '%z' "$MODEL")
if (( MODEL_BYTES >= LARGE_MODEL_BYTES )); then
    PAGE_BYTES=$(vm_stat | awk -F'of ' '/page size of/ {
        gsub(/[^0-9]/, "", $2); print $2; exit
    }')
    COMPRESSOR_PAGES=$(vm_stat | awk -F: '/Pages occupied by compressor/ {
        gsub(/[^0-9]/, "", $2); print $2; exit
    }')
    SWAP_USED_MIB=$(sysctl -n vm.swapusage | awk '{
        for (i = 1; i <= NF; i++) {
            if ($i == "used") {
                value = $(i + 2)
                gsub(/M/, "", value)
                printf "%.0f\n", value
                exit
            }
        }
    }')
    read -r LARGEST_PID LARGEST_RSS_KIB LARGEST_COMMAND < <(
        ps -axo pid=,rss=,comm= | awk -v self="$$" '$1 != self && $2 > max {
            max = $2; pid = $1; command = $3
        } END { print pid + 0, max + 0, command }'
    )
    [[ -n "$PAGE_BYTES" && -n "$COMPRESSOR_PAGES" && -n "$SWAP_USED_MIB" ]] || {
        echo "could not read macOS memory pressure counters" >&2
        exit 4
    }

    COMPRESSOR_BYTES=$((COMPRESSOR_PAGES * PAGE_BYTES))
    SWAP_USED_BYTES=$((SWAP_USED_MIB * 1024 * 1024))
    LARGEST_RSS_BYTES=$((LARGEST_RSS_KIB * 1024))
    MAX_SWAP_USED_BYTES=$((MAX_SWAP_USED_GIB * 1024 * 1024 * 1024))
    MAX_COMPRESSOR_USED_BYTES=$((MAX_COMPRESSOR_USED_GIB * 1024 * 1024 * 1024))
    MAX_OTHER_PROCESS_RSS_BYTES=$((MAX_OTHER_PROCESS_RSS_GIB * 1024 * 1024 * 1024))

    printf 'hf2q memory preflight: model=%.2f GiB swap_used=%.2f GiB compressor=%.2f GiB largest_process=%.2f GiB (%s pid %s)\n' \
        "$(awk -v bytes="$MODEL_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$SWAP_USED_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$COMPRESSOR_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$LARGEST_RSS_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$LARGEST_COMMAND" "$LARGEST_PID"

    MEMORY_FAILURES=()
    if (( SWAP_USED_BYTES > MAX_SWAP_USED_BYTES )); then
        MEMORY_FAILURES+=("swap use exceeds ${MAX_SWAP_USED_GIB} GiB")
    fi
    if (( COMPRESSOR_BYTES > MAX_COMPRESSOR_USED_BYTES )); then
        MEMORY_FAILURES+=("compressor use exceeds ${MAX_COMPRESSOR_USED_GIB} GiB")
    fi
    if (( LARGEST_RSS_BYTES > MAX_OTHER_PROCESS_RSS_BYTES )); then
        MEMORY_FAILURES+=("process $LARGEST_COMMAND (pid $LARGEST_PID) exceeds ${MAX_OTHER_PROCESS_RSS_GIB} GiB RSS")
    fi
    if (( ${#MEMORY_FAILURES[@]} > 0 )); then
        printf 'unsafe host memory state for this model:\n' >&2
        printf '  - %s\n' "${MEMORY_FAILURES[@]}" >&2
        if (( UNSAFE_MEMORY_OVERRIDE == 0 )); then
            echo "refusing before model load; close the named workload or reboot to clear stale paging state" >&2
            echo "for a controlled diagnostic only: HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE=1 $0" >&2
            exit 4
        fi
        echo "WARNING: proceeding despite unsafe host memory state" >&2
    fi
fi

if (( CHECK_ONLY == 1 )); then
    echo "hf2q DeepSeek-V4 preflight passed; no model was loaded"
    exit 0
fi

exec env \
    HF2Q_DEEPSEEK_MAX_SEQ_LEN="$CONTEXT_LEN" \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    "$HF2Q_BIN" -v serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --overflow-policy reject \
        --scheduler fifo-serial
