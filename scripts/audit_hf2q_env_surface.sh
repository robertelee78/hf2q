#!/usr/bin/env bash
# Emit the complete lexical HF2Q_* inventory under src/ with the product
# disposition accepted in ADR-050. This is intentionally a lexical audit:
# names in regression tests and documentation strings are included so a
# removed production reader cannot disappear from review.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

classify() {
    case "$1" in
        HF2Q_SCHEDULER|HF2Q_MAX_SLOTS|HF2Q_KV_PERSIST_BUDGET_BYTES)
            printf '%s' 'removed-guard'
            ;;
        HF2Q_AUTH_TOKEN|HF2Q_CACHE_DIR|HF2Q_NO_COMPLETION_INSTALL|\
        HF2Q_COMPLETION_STARTUP_FILE|HF2Q_FISH_COMPLETIONS_DIR|\
        HF2Q_ZSH_COMPLETIONS_DIR|HF2Q_ZSH_STARTUP_DIR)
            printf '%s' 'appropriate-env'
            ;;
        HF2Q_KV_PERSIST|HF2Q_KV_PERSIST_PATH|\
        HF2Q_KV_LCP_CAPACITY|HF2Q_KV_LCP_CHUNKED_PREFILL|\
        HF2Q_KV_LCP_LONG_RESUME|HF2Q_KV_LCP_RESUME|\
        HF2Q_KV_LCP_RESUME_CAPACITY|HF2Q_POOL_BUDGET_BYTES|\
        HF2Q_MAX_BATCHED_SLOTS|HF2Q_SPEC_DECODE_MAX_BATCHED_SLOTS|\
        HF2Q_CROSS_SLOT_ADMIT|HF2Q_ADMIT_COALESCE_US|\
        HF2Q_PREFILL_CROSS_SLOT|HF2Q_PREFILL_SLOT_BATCHED|\
        HF2Q_SERVE_BATCHED|HF2Q_SERVE_BATCHED_PREFILL|\
        HF2Q_TQ_KV|HF2Q_HYBRID_KV|HF2Q_ENCODER_SESSION|\
        HF2Q_FFN_TERMINAL_K_BATCH|HF2Q_LMHEAD_Q6K|HF2Q_QWEN_SPECULATION|\
        HF2Q_DECODE_MVN|HF2Q_DECODE_MV_EXT)
            printf '%s' 'promote-or-internalize'
            ;;
        HF2Q_LMHEAD_Q8|HF2Q_QWEN_GQA_Q2|HF2Q_BATCHED_PREFILL|\
        HF2Q_STREAMING_PHASE3)
            printf '%s' 'documented-escape'
            ;;
        *)
            printf '%s' 'development-only'
            ;;
    esac
}

printf '%s\n' '| Name | Disposition | Source occurrences |'
printf '%s\n' '|---|---|---:|'
while IFS= read -r name; do
    count=$(rg -o "$name" src --glob '*.rs' | wc -l | tr -d ' ')
    printf '| `%s` | `%s` | %s |\n' "$name" "$(classify "$name")" "$count"
done < <(rg -o 'HF2Q_[A-Z0-9_]+' src --glob '*.rs' \
    | sed 's/.*://' \
    | sort -u)
