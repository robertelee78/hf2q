#!/usr/bin/env bash
# serve_qwen36_opencode.sh — canonical hf2q serve launcher for the
# Qwen 3.6 35B-A3B APEX GGUF, tuned for opencode agentic coding.
#
# Every flag/env below is load-bearing and verified 2026-08-03 on M5 Max
# (see docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md sub-iter
# 23d-γ for the prefix-cache coherence gates):
#
#   HF2Q_QWEN36_AUTOREG=1   REQUIRED — qwen3.6 GGUFs are gated behind the
#                           Wave-5a autoregressive opt-in (serve refuses
#                           to load the model without it).
#   HF2Q_KV_LCP_RESUME=1    Enables LCP partial-prefill resume. Strictly
#                           OPTIONAL post-23d-γ (2026-08-03): qwen35's TQ
#                           restore path is proven, so the widened
#                           effective_kv_lcp_resume gate admits this arch
#                           by default. Kept explicit here for
#                           discoverability + older binaries.
#   HF2Q_KV_PERSIST=<dir>   Binds the Qwen35DiskPersistor (LCP snapshots
#                           write through to disk; cold restarts hydrate).
#   HF2Q_KV_PERSIST_BUDGET_BYTES
#                           On-disk budget with LRU eviction (12.8 GiB =
#                           ADR-017 §R-F5's 10%-of-RAM guidance on 128 GB).
#                           Enforced by Qwen35DiskPersistor per cfg subdir
#                           (23d-γ — pre-fix a single ~100K-token opencode
#                           session wrote 105 GB unbudgeted).
#   HF2Q_KV_LCP_DISABLE_MID_STORE=1
#                           Serial agentic sessions retain one compact
#                           latest-turn checkpoint at the verified Qwen
#                           ChatML generation boundary. Intermediate stride
#                           snapshots are redundant for that workload and are
#                           disabled by default. Set MID_STORES=1 only for a
#                           branch-heavy workload that needs older checkpoints.
#   HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=4096
#                           Granularity of optional intermediate checkpoints
#                           when MID_STORES=1 (must be a multiple of 64).
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
#   --kv-persist <dir>      Wires the hot-swap spiller substrate. Set to
#                           the SAME dir as HF2Q_KV_PERSIST (they are two
#                           separate mechanisms; see operating-kv-cache.md).
#   --overflow-policy reject
#                           opencode manages its own compaction; the
#                           server must 400 on overflow (OpenAI semantics)
#                           instead of silently rewriting the conversation
#                           (default "summarize") or truncating it.
#   --scheduler fifo-serial Production default. Per-stream decode stays
#                           at full speed (~123 tok/s); concurrent
#                           requests queue (cap 32 → 429 + Retry-After).
#                           inflight-batched is NOT recommended here:
#                           qwen35moe multi-slot requires HF2Q_TQ_KV=0
#                           (3.94× more KV memory) and slot-aware mode has
#                           no LCP prefix resume (ADR-040 iter-LCP is
#                           structural N/A) — a strict downgrade for
#                           single-user agentic coding.
#
# Family contract and prefix-cache stack:
#   * The loader uses the GGUF-embedded Qwen ChatML template and validates its
#     native turn, tool-call, and tool-response markers before inference.
#   1. HybridPromptCache — exact-repeat requests replay in ~2 ms
#      (greedy-only, single-slot).
#   2. The compact latest-turn checkpoint preserves the stable ChatML prefix
#      immediately before Qwen's temporary generation seed. A 119,728-token
#      continuation reused 119,669 tokens and reached semantic output in
#      0.858 s on the target M5 Max.
#   3. Disk persistence — compact checkpoints survive server restarts
#      across ordinary per-turn cache-capacity changes (QH35 codec v5,
#      capacity-independent + substrate-namespaced fingerprint).
#
# Usage:
#   scripts/serve_qwen36_opencode.sh            # foreground (default)
#   PORT=8082 scripts/serve_qwen36_opencode.sh  # override port
set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
KV_DIR="${KV_DIR:-$HOME/.cache/hf2q/kv-persist}"
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-13743895347}"  # 12.8 GiB
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
MID_STORES="${MID_STORES:-0}"

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 3; }
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not found: $HF2Q_BIN (cargo build --release)" >&2; exit 3; }
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "PORT must be an integer from 1 through 65535 (got: $PORT)" >&2
    exit 3
fi
if [[ "$MID_STORES" != "0" && "$MID_STORES" != "1" ]]; then
    echo "MID_STORES must be 0 or 1 (got: $MID_STORES)" >&2
    exit 3
fi
if [[ "$MID_STORES" == "1" ]]; then
    DISABLE_MID_STORE=0
else
    DISABLE_MID_STORE=1
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

mkdir -p "$KV_DIR"

exec env \
    HF2Q_QWEN36_AUTOREG=1 \
    HF2Q_KV_LCP_RESUME=1 \
    HF2Q_KV_LCP_DISABLE_MID_STORE="$DISABLE_MID_STORE" \
    HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE="${STRIDE:-4096}" \
    HF2Q_KV_PERSIST="$KV_DIR" \
    HF2Q_KV_PERSIST_BUDGET_BYTES="$KV_BUDGET_BYTES" \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    HF2Q_ENCODER_SESSION=1 \
    HF2Q_FFN_TERMINAL_K_BATCH=8 \
    "$HF2Q_BIN" -v serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --kv-persist "$KV_DIR" \
        --overflow-policy reject \
        --scheduler fifo-serial
