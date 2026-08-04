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
#   HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=4096
#                           Checkpoint granularity (must be ×64). Default
#                           1024 snapshots every 1024 tokens; at ~100K
#                           context each snapshot is ~600 MB (full-attn
#                           TQ grows with position), so prefill spent
#                           ~1.2 GB of snapshot+disk I/O per 1024
#                           tokens. 4096 quarters that cost; the resume
#                           granularity loss is ≤4095 tokens (~2 s at
#                           ~2K tok/s) on a boundary miss — the right
#                           trade for long agentic sessions.
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
# Prefix-cache stack (all engaged by this config):
#   1. HybridPromptCache — exact-repeat requests replay in ~2 ms
#      (greedy-only, single-slot).
#   2. LCP partial-prefill resume — shared-prefix requests (every
#      agentic turn) restore stride-1024 checkpoints; measured 9.0×
#      TTFT on a 2.7K-token prompt (2920 ms → 326 ms).
#   3. Disk persistence — checkpoints survive server restarts
#      (QH35 codec v4, substrate-namespaced fingerprint).
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

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 3; }
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not found: $HF2Q_BIN (cargo build --release)" >&2; exit 3; }
mkdir -p "$KV_DIR"

# One-model-at-a-time guard (feedback_oom_prevention): a 35B-class model
# holds ~30 GB of unified memory; a second concurrent inference process
# on the same box risks OOM.
if pgrep -x hf2q >/dev/null 2>&1; then
    echo "another hf2q process is already running — refusing to start a second" >&2
    echo "(kill it first: pkill -x hf2q)" >&2
    exit 1
fi

exec env \
    HF2Q_QWEN36_AUTOREG=1 \
    HF2Q_KV_LCP_RESUME=1 \
    HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE="${STRIDE:-4096}" \
    HF2Q_KV_PERSIST="$KV_DIR" \
    HF2Q_KV_PERSIST_BUDGET_BYTES="$KV_BUDGET_BYTES" \
    HF2Q_DEFAULT_REPETITION_PENALTY="${REP_PENALTY:-1.05}" \
    "$HF2Q_BIN" serve \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --kv-persist "$KV_DIR" \
        --overflow-policy reject \
        --scheduler fifo-serial
