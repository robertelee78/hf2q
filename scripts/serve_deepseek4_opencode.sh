#!/usr/bin/env bash
# serve_deepseek4_opencode.sh — canonical hf2q serve launcher for the
# DeepSeek-V4-Flash-0731 agentic Q2/Q3/Q8 GGUF, tuned for OpenCode coding.
# Pair this server with explicit OpenCode build/plan settings
# `temperature=0.55`, `top_p=0.95`, and a `max` model variant whose
# `reasoningEffort` is `max`; see README.md and the checked-in OpenCode gate.
#
# The DeepSeek cache is native and in-memory:
#
#   --ctx 262144           Advertises a 256K serving limit per slot while initially
#                           allocating a 131K live cache (~877 MiB). Capacity
#                           grows in 131K steps only when the transcript needs
#                           it; the prompt-tail checkpoint stays ~17 MiB.
#                           Growing transcripts prefill only their
#                           suffix; reasoning canonicalization restores the
#                           checkpoint and replays the rewritten tail.
#   --default-repetition-penalty 1.0
#                           No hidden repetition penalty. The previous 1.05
#                           default distorted constrained tool strings and did
#                           not prevent client-side action loops. Operators may
#                           set REP_PENALTY only for a measured workload;
#                           non-default request values still win.
#   --default-tool-thinking-token-budget 8
#                           Bounds forced-open reasoning for the narrow
#                           single-tool required/named-tool path before the
#                           constrained DSML tool call. Set
#                           REQUIRED_TOOL_THINKING_TOKEN_BUDGET=0 to disable
#                           the operator default; explicit request budgets are
#                           not accepted for DeepSeek-V4.
#   HF2Q_DEEPSEEK_PREFILL_WINDOWS=adaptive
#                           Uses the measured 2,048-token transaction while
#                           the live cache is 131K, balances a cold prompt that
#                           would otherwise leave a severely underfilled third
#                           transaction, then uses 1,024 after cache capacity
#                           grows. Set PREFILL_WINDOWS explicitly only when
#                           benchmarking a measured alternative.
#   --overflow-policy reject
#                           OpenCode manages conversation compaction, so the
#                           server reports context overflow instead of silently
#                           rewriting or truncating the transcript.
#   --scheduler inflight-batched --max-slots 4
#                           Four independent agent sessions make progress.
#                           Every slot advertises 262144 tokens; context is never
#                           divided by slot count. MAX_SLOTS=8 is the np8-like
#                           operator setting after the hardware gate passes.
#   --kv-cache-budget
#                           Shared physical KV high-water across those full-
#                           context slots. This bounds residency; it is not a
#                           per-slot logical context limit.
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
#   scripts/serve_deepseek4_opencode.sh --ctx 131072  # lower-memory preset
#   scripts/serve_deepseek4_opencode.sh --ctx 1048576 # full trained window
#   CHECK_ONLY=1 scripts/serve_deepseek4_opencode.sh        # preflight only
#   MAX_SLOTS=8 scripts/serve_deepseek4_opencode.sh         # np8-like
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/hf2q_process_guard.sh
source "$SCRIPT_DIR/hf2q_process_guard.sh"
# shellcheck source=scripts/hf2q_q5_policy.sh
source "$SCRIPT_DIR/hf2q_q5_policy.sh"
hf2q_resolve_q5k_canonical_policy

# The default is the schema-v2, source-bound reproduction used by the strict
# coherence/performance gate. Operators may still set MODEL to any explicitly
# supported DeepSeek-V4 GGUF; serving is not restricted by producer identity.
MODEL="${MODEL:-/opt/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-agentic-q2.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
CONTEXT_TOKENS=262144
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
CHECK_ONLY="${CHECK_ONLY:-0}"
MAX_SLOTS="${MAX_SLOTS:-4}"
KV_CACHE_BUDGET_BYTES="${KV_CACHE_BUDGET_BYTES:-8589934592}" # 8 GiB shared
REQUIRED_TOOL_THINKING_TOKEN_BUDGET="${REQUIRED_TOOL_THINKING_TOKEN_BUDGET:-8}"
MIXED_COHORT="${HF2Q_DEEPSEEK_MIXED_COHORT:-1}"

# Keep the wrapper's context override identical to the real hf2q flag. The
# qualified launcher defaults to 262144, while direct `hf2q serve` omits
# `--ctx` and therefore uses the GGUF maximum.
case "${1:-}" in
    --ctx)
        [[ $# -ge 2 ]] || { echo "--ctx requires a token count" >&2; exit 2; }
        CONTEXT_TOKENS=$2
        shift 2
        ;;
    --ctx=*)
        CONTEXT_TOKENS=${1#--ctx=}
        shift
        ;;
esac
if (( $# != 0 )); then
    echo "unsupported launcher argument: $1 (supported: --ctx TOKENS)" >&2
    exit 2
fi

# A ~100 GiB resident model leaves little margin on a 128 GiB host. Refuse a
# load when prior inference has left the compressor/swap saturated or one
# competing process already consumes a material part of that margin. macOS
# RSS includes reclaimable WebKit and IOAccelerator mappings, so an RSS value
# above the ceiling is refined with `footprint` when available; failure to
# obtain a physical footprint keeps the conservative RSS value. These limits
# are intentionally conservative; the explicit unsafe override exists for
# controlled diagnostics, not normal OpenCode serving.
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
if ! [[ "$CONTEXT_TOKENS" =~ ^[0-9]+$ ]] || (( CONTEXT_TOKENS < 128 )); then
    echo "--ctx must be an integer of at least 128 for DeepSeek-V4 (got: $CONTEXT_TOKENS)" >&2
    exit 3
fi
for SETTING in CHECK_ONLY MAX_SWAP_USED_GIB MAX_COMPRESSOR_USED_GIB \
    MAX_OTHER_PROCESS_RSS_GIB UNSAFE_MEMORY_OVERRIDE MAX_SLOTS \
    KV_CACHE_BUDGET_BYTES REQUIRED_TOOL_THINKING_TOKEN_BUDGET; do
    VALUE=${!SETTING}
    if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
        echo "$SETTING must be a non-negative integer (got: $VALUE)" >&2
        exit 3
    fi
done
if [[ "$MIXED_COHORT" != 0 && "$MIXED_COHORT" != 1 ]]; then
    echo "HF2Q_DEEPSEEK_MIXED_COHORT must be 0 or 1 (got: $MIXED_COHORT)" >&2
    exit 3
fi
if (( CHECK_ONLY > 1 || UNSAFE_MEMORY_OVERRIDE > 1 )); then
    echo "CHECK_ONLY and HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE must be 0 or 1" >&2
    exit 3
fi
if (( MAX_SLOTS < 1 || MAX_SLOTS > 8 )); then
    echo "MAX_SLOTS must be from 1 through 8 (got: $MAX_SLOTS)" >&2
    exit 3
fi
if [[ -n "${PREFILL_WINDOWS:-}" ]]; then
    if ! [[ "$PREFILL_WINDOWS" =~ ^[1-9][0-9]*$ ]]; then
        echo "PREFILL_WINDOWS must be a positive integer (got: $PREFILL_WINDOWS)" >&2
        exit 3
    fi
    export HF2Q_DEEPSEEK_PREFILL_WINDOWS="$PREFILL_WINDOWS"
fi
# Resolve this once before the worker starts. The default is the accepted B.1
# candidate; zero is the exact same-binary serial Mixed control used by the
# fail-closed hardware gate. The worker repeats the resolved value in its
# startup receipt so a dead or misspelled launcher setting cannot pass.
export HF2Q_DEEPSEEK_MIXED_COHORT="$MIXED_COHORT"

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
RUNTIME_PIDS="$(hf2q_active_serve_pids)"
if [[ -n "$RUNTIME_PIDS" ]]; then
    echo "another hf2q server is already running — refusing before model load" >&2
    echo "pid(s): ${RUNTIME_PIDS//$'\n'/, }" >&2
    echo "stop that server before starting DeepSeek-V4" >&2
    exit 1
fi

MODEL_BYTES=$(stat -f '%z' "$MODEL" 2>/dev/null || stat -c '%s' "$MODEL")
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
    [[ -n "$PAGE_BYTES" && -n "$COMPRESSOR_PAGES" && -n "$SWAP_USED_MIB" ]] || {
        echo "could not read macOS memory pressure counters" >&2
        exit 4
    }

    COMPRESSOR_BYTES=$((COMPRESSOR_PAGES * PAGE_BYTES))
    SWAP_USED_BYTES=$((SWAP_USED_MIB * 1024 * 1024))
    MAX_SWAP_USED_BYTES=$((MAX_SWAP_USED_GIB * 1024 * 1024 * 1024))
    MAX_COMPRESSOR_USED_BYTES=$((MAX_COMPRESSOR_USED_GIB * 1024 * 1024 * 1024))
    MAX_OTHER_PROCESS_RSS_BYTES=$((MAX_OTHER_PROCESS_RSS_GIB * 1024 * 1024 * 1024))

    LARGEST_PID=0
    LARGEST_PROCESS_BYTES=0
    LARGEST_PROCESS_METRIC="RSS upper bound"
    LARGEST_COMMAND="unknown"
    while read -r PROCESS_PID PROCESS_RSS_KIB PROCESS_COMMAND; do
        (( PROCESS_PID == $$ )) && continue
        PROCESS_BYTES=$((PROCESS_RSS_KIB * 1024))
        PROCESS_METRIC="RSS upper bound"
        if (( PROCESS_BYTES > MAX_OTHER_PROCESS_RSS_BYTES )) && command -v footprint >/dev/null 2>&1; then
            PHYSICAL_BYTES=$(footprint -p "$PROCESS_PID" -f bytes --noCategories 2>/dev/null | awk '
                $1 == "phys_footprint:" && $2 ~ /^[0-9]+$/ { print $2; exit }
            ' || true)
            if [[ "$PHYSICAL_BYTES" =~ ^[0-9]+$ ]]; then
                PROCESS_BYTES=$PHYSICAL_BYTES
                PROCESS_METRIC="physical footprint"
            fi
        fi
        if (( PROCESS_BYTES > LARGEST_PROCESS_BYTES )); then
            LARGEST_PID=$PROCESS_PID
            LARGEST_PROCESS_BYTES=$PROCESS_BYTES
            LARGEST_PROCESS_METRIC=$PROCESS_METRIC
            LARGEST_COMMAND=$PROCESS_COMMAND
        fi
    done < <(ps -axo pid=,rss=,comm=)

    printf 'hf2q memory preflight: model=%.2f GiB swap_used=%.2f GiB compressor=%.2f GiB largest_process=%.2f GiB (%s pid %s, %s)\n' \
        "$(awk -v bytes="$MODEL_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$SWAP_USED_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$COMPRESSOR_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$(awk -v bytes="$LARGEST_PROCESS_BYTES" 'BEGIN { print bytes / 1073741824 }')" \
        "$LARGEST_COMMAND" "$LARGEST_PID" "$LARGEST_PROCESS_METRIC"

    MEMORY_FAILURES=()
    if (( SWAP_USED_BYTES > MAX_SWAP_USED_BYTES )); then
        MEMORY_FAILURES+=("swap use exceeds ${MAX_SWAP_USED_GIB} GiB")
    fi
    if (( COMPRESSOR_BYTES > MAX_COMPRESSOR_USED_BYTES )); then
        MEMORY_FAILURES+=("compressor use exceeds ${MAX_COMPRESSOR_USED_GIB} GiB")
    fi
    if (( LARGEST_PROCESS_BYTES > MAX_OTHER_PROCESS_RSS_BYTES )); then
        MEMORY_FAILURES+=("process $LARGEST_COMMAND (pid $LARGEST_PID) exceeds ${MAX_OTHER_PROCESS_RSS_GIB} GiB by $LARGEST_PROCESS_METRIC")
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
    env HF2Q_Q5K_CANONICAL_Q4X4="$HF2Q_Q5K_CANONICAL_Q4X4" \
        "$HF2Q_BIN" info \
        --model "$MODEL" \
        --ctx "$CONTEXT_TOKENS" \
        --scheduler inflight-batched \
        --max-slots "$MAX_SLOTS" \
        --kv-cache-budget "$KV_CACHE_BUDGET_BYTES"
    echo "hf2q DeepSeek-V4 host and static serving preflight passed; no tensor payload was loaded"
    exit 0
fi

exec env HF2Q_Q5K_CANONICAL_Q4X4="$HF2Q_Q5K_CANONICAL_Q4X4" \
    "$HF2Q_BIN" -v serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx "$CONTEXT_TOKENS" \
        --overflow-policy reject \
        --scheduler inflight-batched \
        --max-slots "$MAX_SLOTS" \
        --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" \
        --default-repetition-penalty "${REP_PENALTY:-1.0}" \
        --default-tool-thinking-token-budget "$REQUIRED_TOOL_THINKING_TOKEN_BUDGET"
