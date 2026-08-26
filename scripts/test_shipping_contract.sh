#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
contract="$repo_root/docs/shipping-contract.md"
readme="$repo_root/README.md"
manifest="$repo_root/Cargo.toml"
lockfile="$repo_root/Cargo.lock"
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
mlx_native_version="$(
    awk '
        /^\[/ {
            in_mlx_table = ($0 == "[dependencies.mlx-native]")
            if (in_mlx_table) table_count++
            next
        }
        !in_mlx_table && /^mlx-native[[:space:]]*=/ {
            direct_count++
            line = $0
            sub(/^mlx-native[[:space:]]*=[[:space:]]*"=/, "", line)
            sub(/"[[:space:]]*$/, "", line)
            if (line == $0) invalid = 1
            direct_version = line
            next
        }
        in_mlx_table && /^[[:space:]]*(#.*)?$/ { next }
        in_mlx_table && /^version[[:space:]]*=/ {
            table_version_count++
            line = $0
            sub(/^version[[:space:]]*=[[:space:]]*"=/, "", line)
            sub(/"[[:space:]]*$/, "", line)
            if (line == $0) invalid = 1
            table_version = line
            next
        }
        in_mlx_table { invalid = 1 }
        END {
            declaration_count = direct_count + table_count
            if (invalid || declaration_count != 1) exit 1
            if (direct_count == 1 && table_count == 0) {
                print direct_version
                exit
            }
            if (table_count == 1 && direct_count == 0 && table_version_count == 1) {
                print table_version
                exit
            }
            exit 1
        }
    ' "$manifest"
)" || fail "Cargo.toml must contain exactly one exact registry mlx-native dependency"
[[ "$mlx_native_version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] \
    || fail "Cargo.toml mlx-native requirement is not exact stable SemVer"
locked_mlx_native_identity="$(
    awk '
        function finish_package() {
            if (!in_package || name != "mlx-native") return
            mlx_count++
            if (version == "" || source == "" || checksum == "") invalid = 1
            mlx_version = version
            mlx_source = source
            mlx_checksum = checksum
        }
        $0 == "[[package]]" {
            finish_package()
            in_package = 1
            name = version = source = checksum = ""
            next
        }
        in_package && /^name = / { name = $3; gsub(/\"/, "", name); next }
        in_package && /^version = / { version = $3; gsub(/\"/, "", version); next }
        in_package && /^source = / { source = $3; gsub(/\"/, "", source); next }
        in_package && /^checksum = / { checksum = $3; gsub(/\"/, "", checksum); next }
        END {
            finish_package()
            if (invalid || mlx_count != 1) exit 1
            print mlx_version, mlx_source, mlx_checksum
        }
    ' "$lockfile"
)" || fail "Cargo.lock must contain exactly one complete mlx-native registry record"
read -r locked_mlx_native_version locked_mlx_native_source locked_mlx_native_checksum \
    <<<"$locked_mlx_native_identity"
[[ "$locked_mlx_native_version" == "$mlx_native_version" ]] \
    || fail "Cargo.toml and Cargo.lock mlx-native versions differ"
[[ "$locked_mlx_native_source" == registry+https://github.com/rust-lang/crates.io-index ]] \
    || fail "Cargo.lock mlx-native source is not crates.io"
[[ "$locked_mlx_native_checksum" =~ ^[0-9a-f]{64}$ ]] \
    || fail "Cargo.lock mlx-native checksum is missing or malformed"

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
readme_status_count="$(grep -Fc '| **Status** |' "$readme")"
[[ "$readme_status_count" == 1 ]] || fail "README must contain exactly one Status row"
readme_status="$(grep -F '| **Status** |' "$readme")"
[[ "$readme_status" == *"mlx-native ${mlx_native_version}"* \
    && "$readme_status" == *"$locked_mlx_native_checksum"* ]] \
    || fail "README Status row differs from the exact mlx-native lock identity"
shipping_header="$(sed -n '1,16p' "$contract")"
[[ "$shipping_header" == *"mlx-native = ${mlx_native_version}"* \
    && "$shipping_header" == *"$locked_mlx_native_checksum"* ]] \
    || fail "shipping-contract candidate stanza differs from the exact mlx-native lock identity"
require_literal "$contract" "### Supported family and command matrix"
require_literal "$contract" "| Qwen3.5 / Qwen3.6"
require_literal "$contract" "| Qwen3.8-27B"
require_literal "$contract" "| BERT / Nomic-BERT"
require_literal "$contract" "| Llama 3 / MiniMax M2.7"
require_literal "$contract" "| Qwen3.5 / Qwen3.6 / Qwen3.8 |"
require_literal "$contract" "\`HF2Q_DECODE_MVN\`"
require_literal "$contract" "\`HF2Q_DECODE_MV_EXT\`"

reject_literal "$contract" "Qwen SlotAware soft-token, deepstack, and 3D-position generation"
reject_literal "$readme" "Standalone Qwen3-VL"

# The canonical DeepSeek launcher and matched peer benchmark share the real
# typed context surface. A benchmark-local CONTEXT_LEN remains valid input to
# the harness, but it must reach hf2q as --ctx rather than a dead launcher env.
require_literal "$deepseek_launcher" '--ctx "$CONTEXT_TOKENS"'
reject_literal "$deepseek_launcher" 'CONTEXT_LEN'
require_literal "$deepseek_parity" 'serve_deepseek4_opencode.sh" --ctx "$CONTEXT_LEN"'
reject_literal "$deepseek_parity" 'HF2Q_BIN="$HF2Q_BIN" CONTEXT_LEN='

bash "$repo_root/scripts/test_getting_started_guide.sh"

echo "shipping-contract check passed for v${crate_version}"
