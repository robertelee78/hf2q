#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
contract="$repo_root/docs/shipping-contract.md"
readme="$repo_root/README.md"
manifest="$repo_root/Cargo.toml"
arch_catalog="$repo_root/src/quantize/ggml_quants/tensor_ref.rs"
deepseek_launcher="$repo_root/scripts/serve_deepseek4_opencode.sh"
deepseek_parity="$repo_root/scripts/benchmark_deepseek4_server_parity.sh"

fail() {
    echo "shipping-contract check failed: $*" >&2
    exit 1
}

require_literal() {
    local file="$1"
    local literal="$2"
    grep -Fq -- "$literal" "$file" || fail "missing '$literal' in ${file#"$repo_root/"}"
}

reject_literal() {
    local file="$1"
    local literal="$2"
    if grep -Fq -- "$literal" "$file"; then
        fail "stale '$literal' remains in ${file#"$repo_root/"}"
    fi
}

crate_version="$(awk -F '"' '/^version = "/ { print $2; exit }' "$manifest")"
[[ -n "$crate_version" ]] || fail "could not read package version from Cargo.toml"

catalog_variants="$(
    awk '
        /pub enum ArchName/ { in_catalog = 1; next }
        in_catalog && /C-fidelity placeholders/ { exit }
        in_catalog && /^[[:space:]]+[A-Z][A-Za-z0-9]*,/ {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            sub(/,.*/, "", line)
            print line
        }
    ' "$arch_catalog"
)"
expected_variants="$(printf '%s\n' \
    Gemma4 \
    Gemma4Mmproj \
    Gemma4VisionMmproj \
    Qwen35 \
    Qwen35Moe \
    Qwen35MoeFull \
    Qwen3VlText \
    Bert \
    NomicBert \
    Llama3 \
    MiniMaxM2 \
    Deepseek4)"
[[ "$catalog_variants" == "$expected_variants" ]] || {
    printf 'shipping-contract check failed: conversion architecture catalog changed\nexpected:\n%s\nactual:\n%s\n' \
        "$expected_variants" "$catalog_variants" >&2
    exit 1
}

require_literal "$contract" "Current published release: \`v${crate_version}\`"
require_literal "$contract" "### Supported family and command matrix"
require_literal "$contract" "| Qwen3.5 / Qwen3.6"
require_literal "$contract" "| Qwen3.8-27B"
require_literal "$contract" "| Standalone Qwen3-VL"
require_literal "$contract" "| BERT / Nomic-BERT"
require_literal "$contract" "| Llama 3 / MiniMax M2.7"
require_literal "$contract" "| Qwen3.5 / Qwen3.6 / Qwen3.8 |"
require_literal "$contract" "\`HF2Q_DECODE_MVN\`"
require_literal "$contract" "\`HF2Q_DECODE_MV_EXT\`"
require_literal "$contract" "ADR-041"

reject_literal "$contract" "Qwen SlotAware soft-token, deepstack, and 3D-position generation"
reject_literal "$readme" "vision (\`qwen3vl\`)"
require_literal "$readme" "Standalone Qwen3-VL generation and serving fail closed"

# The canonical DeepSeek launcher and matched peer benchmark share the real
# typed context surface. A benchmark-local CONTEXT_LEN remains valid input to
# the harness, but it must reach hf2q as --ctx rather than a dead launcher env.
require_literal "$deepseek_launcher" '--ctx "$CONTEXT_TOKENS"'
reject_literal "$deepseek_launcher" 'CONTEXT_LEN'
require_literal "$deepseek_parity" 'serve_deepseek4_opencode.sh" --ctx "$CONTEXT_LEN"'
reject_literal "$deepseek_parity" 'HF2Q_BIN="$HF2Q_BIN" CONTEXT_LEN='

bash "$repo_root/scripts/test_getting_started_guide.sh"

echo "shipping-contract check passed for v${crate_version}"
