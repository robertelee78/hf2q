#!/usr/bin/env bash
# serve_gemma4_opencode.sh — canonical hf2q serve launcher for the
# Gemma 4 Ara 26B APEX GGUF, tuned for opencode agentic coding.
#
# Verified 2026-08-03 on M5 Max (docs/ADR-017-persistent-block-prefix-cache.md
# "gemma-hybrid-lcp" + long-resume addenda):
#
#   HF2Q_KV_LCP_RESUME=1    LCP partial-prefill resume. Default-on under the
#                           production hybrid regime post-"gemma-hybrid-lcp";
#                           kept explicit for discoverability/older binaries.
#   HF2Q_KV_LCP_LONG_RESUME=1
#                           Extends LCP to prompts > sliding_window (1024) —
#                           sliding layers allocate LINEAR buffers + the
#                           hybrid SDPA kernel applies bounded chronological
#                           staging (byte-identity-gated vs a cold request
#                           through the same production graph
#                           in tests/lcp_partial_prefill_byte_identity.rs::
#                           gemma_hybrid_long_resume_byte_identity).
#   HF2Q_KV_LCP_RESUME_CAPACITY=8g
#                           Registry budget for short or branched-prefix
#                           snapshots. The normal fifo-serial continuation
#                           reuses Gemma's already-resident dense+hybrid KV in
#                           place and does not duplicate a long live prefix.
#   --mmproj                Gemma 4 vision tower (optional; enables image
#                           parts in chat completions). Delete the flag for
#                           a text-only server.
#   --overflow-policy reject
#                           opencode manages its own compaction; the server
#                           must 400 on overflow (OpenAI semantics).
#   --scheduler inflight-batched --max-slots 4
#                           Four independent agent sessions make progress.
#                           Every slot retains Gemma's full model context;
#                           neither context nor KV capacity is divided by N.
#   --kv-cache-budget-bytes Shared physical high-water across full-context
#                           slots. MAX_SLOTS=8 is the np8-like setting.
#   HF2Q_ADMIT_COALESCE_US=25000
#                           When every slot is idle, wait at most 25 ms for
#                           peer agent requests so their appended suffixes can
#                           share one transformer-body pass. Active decode is
#                           never delayed for collection.
#   HF2Q_DEFAULT_REPETITION_PENALTY=1.05
#                           Loop mitigation (2026-08-03), same knob as
#                           serve_qwen36_opencode.sh — opencode cannot send
#                           repetition_penalty, so omission-clients sampled
#                           with 1.0 and long sessions looped (the garbage
#                           then re-primed via compacted history). Applied
#                           only when the client omits the param (explicit
#                           values win), only to generated tokens (never the
#                           prompt), never to the T=0 GPU argmax path.
#
# FAMILY TEMPLATE AND PREFILL ROUTING:
#   The loader uses the GGUF-embedded Gemma APEX template and validates its
#   native turn, tool-call, and tool-response markers before inference.
#
#   Short cold prompts still auto-route through the existing eager batched
#   path when its O(n²) overhead fits 1/6 of currently available RAM.
#   Overhead is config-dependent:
#     default (tensor-mm globals): ~72 B/seq² — masks + pf_kq scratch
#       ⇒ batched envelope ≈ ≤12K tokens on a 128 GB box
#     HF2Q_GLOBAL_FA=1 (FA globals): ~8 B/seq² — masks only
#       ⇒ batched envelope ≈ ≤35-40K tokens
#   Long plain-text SlotAware prompts instead install resumable states. Each
#   Metal transaction contains at most 4,096 aggregate rows; compatible
#   installed lanes may divide those rows (four equal lanes advance 1,024
#   rows each) without multiplying the watchdog boundary by MAX_SLOTS.
#   Long soft-token work remains fail-closed until it has a resumable graph.
#
#   Normal follow-ups use zero-copy live-prefix reuse and process only the
#   uncached suffix in bounded 256-query chunks. A real 6,784-token OpenCode
#   tool turn reused 6,780 tokens and completed a 6,042-token tool-result
#   suffix at about 909 tok/s with the exact required answer.
#   Env escapes (neither should be necessary):
#     HF2Q_SERVE_BATCHED_PREFILL=0  force linear route always
#     HF2Q_SERVE_BATCHED_PREFILL=1  force batched route always (can
#         reproduce the 2026-08-03 command-buffer OOM — diagnostic only)
#     BATCHED=0 (this script)        disables both SerialFifo and slot-aware
#                                    batched prefill for parity diagnostics
#
# NOT enabled (documented follow-ups, do NOT turn on blindly):
#   * Disk persistence of the LCP registry for gemma (ADR-017 spiller
#     family) — the gemma spill codecs were written for the dense/HB
#     regimes; hybrid-regime (F16-K + TQ-HB V) spill needs its own audit
#     + byte-identity gate. In-memory LCP only: server restarts lose the
#     cache (prefill cost only, never a correctness event).
#   * HF2Q_F16_KV=1 — the F16 dense-KV opt-in has a KNOWN regression vs
#     F32 on gemma4 (ADR-009: sourdough 3656→3095). Keep F32 dense.
#
# Prefix-cache stack:
#   1. PromptCache — exact-repeat requests replay instantly (greedy-only).
#   2. Per-slot live-prefix reuse — each agent's next turn appends to its own
#      resident dense+hybrid KV without displacing another agent.
#   3. LCP registry — short or branched prefixes may resume from dual-leg
#      snapshots when they fit the configured byte budget.
#
# Usage:
#   scripts/serve_gemma4_opencode.sh            # foreground (default)
#   PORT=8086 scripts/serve_gemma4_opencode.sh  # override port
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/models/gemma4/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
MMPROJ="${MMPROJ:-/opt/hf2q/models/gemma4/mmproj-gemma4-f16.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"
LCP_CAPACITY="${LCP_CAPACITY:-8g}"
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

# One-model-at-a-time guard (feedback_oom_prevention): a 26B-class model
# holds ~25-35 GB of unified memory; concurrent inference processes on
# one box OOM it (measured 2026-08-03).
if RUNTIME_PIDS="$(pgrep -x hf2q 2>/dev/null)"; then
    echo "another hf2q server is already running — refusing before model load" >&2
    echo "pid(s): ${RUNTIME_PIDS//$'\n'/, }" >&2
    echo "stop that server before starting Gemma" >&2
    exit 1
fi

# BATCHED unset = engine auto-routes per request (recommended; see
# header). BATCHED=0 forces the linear route; BATCHED=1 forces batched
# (diagnostic only — can reproduce the 92K command-buffer OOM).
# ${BATCHED_ENV:+...} expands to NOTHING when unset/empty, so the env
# var reaches the server only on explicit operator request (macOS bash
# 3.2 + set -u safe: the :+ form never trips unbound-variable).
BATCHED_ENV=""
if [[ "${BATCHED:-}" == "0" ]]; then
    BATCHED_ENV=0
elif [[ "${BATCHED:-}" == "1" ]]; then
    BATCHED_ENV=1
fi

# macOS bash 3.2 + `set -u`: expanding an EMPTY array via "${MMARGS[@]}"
# is an unbound-variable error (this killed the no-mmproj boot
# 2026-08-03). Branch the exec instead of relying on the expansion.
if [[ -f "$MMPROJ" ]]; then
    exec env \
        HF2Q_KV_LCP_RESUME=1 \
        HF2Q_KV_LCP_LONG_RESUME=1 \
        HF2Q_KV_LCP_RESUME_CAPACITY="$LCP_CAPACITY" \
        HF2Q_CROSS_SLOT_ADMIT=1 \
        HF2Q_ADMIT_COALESCE_US="${ADMIT_COALESCE_US:-25000}" \
        ${BATCHED_ENV:+HF2Q_SERVE_BATCHED_PREFILL="$BATCHED_ENV"} \
        ${BATCHED_ENV:+HF2Q_PREFILL_SLOT_BATCHED="$BATCHED_ENV"} \
        HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
        "$HF2Q_BIN" -v serve \
            --model "$MODEL" \
            --mmproj "$MMPROJ" \
            --host "$HOST" \
            --port "$PORT" \
            --overflow-policy reject \
            --scheduler inflight-batched \
            --max-slots "$MAX_SLOTS" \
            --kv-cache-budget-bytes "$KV_CACHE_BUDGET_BYTES"
else
    exec env \
        HF2Q_KV_LCP_RESUME=1 \
        HF2Q_KV_LCP_LONG_RESUME=1 \
        HF2Q_KV_LCP_RESUME_CAPACITY="$LCP_CAPACITY" \
        HF2Q_CROSS_SLOT_ADMIT=1 \
        HF2Q_ADMIT_COALESCE_US="${ADMIT_COALESCE_US:-25000}" \
        ${BATCHED_ENV:+HF2Q_SERVE_BATCHED_PREFILL="$BATCHED_ENV"} \
        ${BATCHED_ENV:+HF2Q_PREFILL_SLOT_BATCHED="$BATCHED_ENV"} \
        HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
        "$HF2Q_BIN" -v serve \
            --model "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --overflow-policy reject \
            --scheduler inflight-batched \
            --max-slots "$MAX_SLOTS" \
            --kv-cache-budget-bytes "$KV_CACHE_BUDGET_BYTES"
fi
