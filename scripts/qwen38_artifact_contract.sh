#!/usr/bin/env bash

# Exact Qwen3.8 artifact identities shared by correctness and performance
# gates. The source repository is immutable at the pinned revision below;
# every gate binds format, byte length, SHA-256, and GGUF file type before it
# may load a model or publish a passing receipt.

# These constants are consumed by callers after sourcing this contract.
# shellcheck disable=SC2034
readonly QWEN38_QUALIFIED_MODEL_REPOSITORY='jenerallee78/Qwen3.8-27B-Abliterated-SFT'
# shellcheck disable=SC2034
readonly QWEN38_QUALIFIED_MODEL_REVISION='0a72776892f98db49381fdf69f4b9982222ec9dc'
# Canonical proof policy. Matrix runners pass these values explicitly and
# outer validators reject both uniformly weakened and cross-format drift.
readonly QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES=51539607552
readonly QWEN38_PHYSICAL_MAX_TOKENS=64
readonly QWEN38_PHYSICAL_DECODE_MVN=1
readonly QWEN38_PHYSICAL_DECODE_MV_EXT=0
readonly QWEN38_PHYSICAL_Q5K_CANONICAL_Q4X4=1
readonly QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS=0.100
# Matched-performance receipts distinguish engine-specific shipping policy
# instead of pretending unlike implementations are one common setting.
readonly QWEN38_MATCHED_HF2Q_SPECULATION_POLICY='adaptive-history-then-mtp-cost-gated'
readonly QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY='fixed-k3-mtp'
readonly QWEN38_MATCHED_HF2Q_KV_CACHE='tq-kv'
readonly QWEN38_MATCHED_REFERENCE_KV_CACHE_K='q8_0'
readonly QWEN38_MATCHED_REFERENCE_KV_CACHE_V='q8_0'
readonly QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4=1
readonly QWEN38_MATCHED_CONTEXT_TOKENS=262144
QWEN38_ARTIFACT_CONTRACT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly QWEN38_ARTIFACT_CONTRACT_ROOT_DIR

qwen38_artifact_formats() {
    printf '%s\n' BF16 Q4_K_M Q5_K_M Q6_K Q8_0
}

qwen38_artifact_snapshot() {
    local artifact_path=$1
    [[ -f "$artifact_path" && -r "$artifact_path" ]] || return 1
    stat -f '%d:%i:%z:%m:%c' "$artifact_path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$artifact_path"
}

qwen38_validate_artifact_snapshot_unchanged() {
    local artifact_path=$1 expected_snapshot=$2 current_snapshot
    [[ -n "$expected_snapshot" ]] || return 1
    current_snapshot=$(qwen38_artifact_snapshot "$artifact_path") || return 1
    [[ "$current_snapshot" == "$expected_snapshot" ]]
}

# Tab-separated fields: format, repository-relative file, bytes, SHA-256,
# GGUF general.file_type.
qwen38_artifact_record() {
    case ${1:-} in
        BF16)
            printf '%s\t%s\t%s\t%s\t%s\n' \
                BF16 gguf/qwen38-abliterated-sft-bf16.gguf \
                54657734208 \
                f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee \
                32
            ;;
        Q4_K_M)
            printf '%s\t%s\t%s\t%s\t%s\n' \
                Q4_K_M gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf \
                16810714944 \
                1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a \
                15
            ;;
        Q5_K_M)
            printf '%s\t%s\t%s\t%s\t%s\n' \
                Q5_K_M gguf/qwen38-abliterated-sft-q5_k_m.gguf \
                19535701568 \
                4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e \
                17
            ;;
        Q6_K)
            printf '%s\t%s\t%s\t%s\t%s\n' \
                Q6_K gguf/qwen38-abliterated-sft-q6_k.gguf \
                22431000128 \
                78f62a87ef851443d4e0c74c4e1eb1dfe73e3bf0ded3cf320ec80f763020ddb3 \
                18
            ;;
        Q8_0)
            printf '%s\t%s\t%s\t%s\t%s\n' \
                Q8_0 gguf/qwen38-abliterated-sft-q8_0.gguf \
                29047084608 \
                53c076e5117be1391e76a9746998fbe2040e6b69a73aa47d1c1b0ca97a8a2c99 \
                7
            ;;
        *)
            echo "Qwen3.8 artifact format must be exactly one of BF16, Q4_K_M, Q5_K_M, Q6_K, or Q8_0" >&2
            return 1
            ;;
    esac
}

qwen38_validate_artifact_identity() {
    local requested_format=$1
    local actual_sha256=$2
    local actual_bytes=$3
    local actual_file_type=$4
    local _format _file expected_bytes expected_sha256 expected_file_type

    IFS=$'\t' read -r _format _file expected_bytes expected_sha256 expected_file_type \
        <<<"$(qwen38_artifact_record "$requested_format")" || return 1
    [[ "$actual_sha256" =~ ^[0-9a-f]{64}$ \
        && "$actual_bytes" =~ ^[0-9]+$ \
        && "$actual_file_type" =~ ^[0-9]+$ ]] || {
        echo "invalid Qwen3.8 artifact identity fields for $requested_format" >&2
        return 1
    }
    [[ "$actual_sha256" == "$expected_sha256" \
        && "$actual_bytes" == "$expected_bytes" \
        && "$actual_file_type" == "$expected_file_type" ]] || {
        echo "Qwen3.8 artifact identity mismatch for $requested_format" >&2
        echo "expected sha256=$expected_sha256 bytes=$expected_bytes file_type=$expected_file_type" >&2
        echo "actual   sha256=$actual_sha256 bytes=$actual_bytes file_type=$actual_file_type" >&2
        return 1
    }
}

qwen38_pinned_peer_pin_path() {
    printf '%s\n' "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/data/llama_cpp_pin.txt"
}

qwen38_pinned_peer_commit() {
    local pin_path
    pin_path=$(qwen38_pinned_peer_pin_path) || return 1
    local pin line_count
    [[ -f "$pin_path" && -r "$pin_path" ]] || {
        echo "pinned peer identity file is missing: $pin_path" >&2
        return 1
    }
    line_count=$(awk 'NF { count++ } END { print count + 0 }' "$pin_path")
    pin=$(awk 'NF { print; exit }' "$pin_path")
    [[ "$line_count" == 1 && "$pin" =~ ^[0-9a-f]{40}$ ]] || {
        echo "pinned peer identity must contain exactly one lowercase commit" >&2
        return 1
    }
    printf '%s\n' "$pin"
}

qwen38_validate_pinned_peer_commit() {
    local actual=$1
    local expected
    expected=$(qwen38_pinned_peer_commit) || return 1
    [[ "$actual" == "$expected" ]] || {
        echo "pinned peer commit mismatch: expected=$expected actual=$actual" >&2
        return 1
    }
}

# Cargo discovers configuration from the source ancestry and Cargo home even
# when those files are ignored by Git. Exact-artifact gates must not inherit a
# path patch, source replacement, compiler wrapper, or other active ambient
# override. Empty and comment-only files are inert and therefore permitted;
# compiler/toolchain environment overrides are rejected separately below.
qwen38_cargo_configuration_is_active() {
    local candidate=$1
    [[ -e "$candidate" || -L "$candidate" ]] || return 1
    [[ -f "$candidate" && -r "$candidate" ]] || return 0
    awk 'BEGIN { active=0 } /^[[:space:]]*(#|$)/ { next } { active=1 } END { exit !active }' \
        "$candidate"
}

qwen38_reject_cargo_configuration() {
    local source_root=$1
    local cargo_home=${2:-${CARGO_HOME:-${HOME:?HOME is required}/.cargo}}
    local current candidate

    source_root=$(cd "$source_root" && pwd -P) || return 1
    current=$source_root
    while :; do
        for candidate in "$current/.cargo/config" "$current/.cargo/config.toml"; do
            if qwen38_cargo_configuration_is_active "$candidate"; then
                echo "exact-artifact gate rejects inherited Cargo configuration: $candidate" >&2
                return 1
            fi
        done
        [[ "$current" == / ]] && break
        current=$(dirname "$current")
    done
    for candidate in "$cargo_home/config" "$cargo_home/config.toml"; do
        if qwen38_cargo_configuration_is_active "$candidate"; then
            echo "exact-artifact gate rejects Cargo-home configuration: $candidate" >&2
            return 1
        fi
    done
    local name value canonical_cargo_home
    canonical_cargo_home=$(cd "$cargo_home" && pwd -P) || return 1
    while IFS='=' read -r name value; do
        case "$name" in
            CARGO_HOME)
                [[ -z "$value" || "$(cd "$value" 2>/dev/null && pwd -P)" == "$canonical_cargo_home" ]] || {
                    echo "exact-artifact gate rejects foreign CARGO_HOME: $value" >&2
                    return 1
                }
                ;;
            CARGO_*|RUST*)
                [[ -z "$value" ]] || {
                    echo "exact-artifact gate rejects ambient build/toolchain override: $name" >&2
                    return 1
                }
                ;;
        esac
    done < <(env)
}

# Run an exact-artifact Rust gate against the source tree named by the receipt,
# independent of the caller's current directory. This prevents a candidate
# runner launched from `/opt/hf2q` (or any other checkout) from silently
# compiling and testing that checkout instead.
qwen38_cargo_test_from_source() {
    local source_root=$1
    shift
    [[ "$source_root" == /* && -f "$source_root/Cargo.toml" ]] || {
        echo "exact gate source root must contain an absolute Cargo.toml: $source_root" >&2
        return 1
    }
    cargo test --manifest-path "$source_root/Cargo.toml" "$@"
}

# Print the exact registry identity of the mlx-native dependency as
# version<TAB>source<TAB>checksum. The manifest requirement is intentionally
# derived instead of duplicating a release-pending checksum constant here.
qwen38_mlx_native_registry_identity() {
    local source_root=$1
    local manifest_versions lock_records record_count
    local manifest_version lock_version lock_source lock_checksum

    [[ -f "$source_root/Cargo.toml" && -f "$source_root/Cargo.lock" ]] || {
        echo "exact-artifact dependency manifests are missing" >&2
        return 1
    }
    manifest_versions=$(sed -nE \
      's/^[[:space:]]*mlx-native[[:space:]]*=[[:space:]]*"=([0-9]+\.[0-9]+\.[0-9]+)"[[:space:]]*$/\1/p' \
      "$source_root/Cargo.toml")
    [[ "$(printf '%s\n' "$manifest_versions" | awk 'NF { count++ } END { print count + 0 }')" == 1 ]] || {
        echo "mlx-native must have exactly one exact manifest requirement" >&2
        return 1
    }
    manifest_version=$(printf '%s\n' "$manifest_versions" | awk 'NF { print; exit }')
    lock_records=$(awk '
      function emit() {
        if (in_package && name == "mlx-native") {
          print version "\t" source "\t" checksum
        }
      }
      /^\[\[package\]\]$/ {
        emit()
        in_package = 1
        name = version = source = checksum = ""
        next
      }
      in_package && /^name = "/ {
        value = $0
        sub(/^name = "/, "", value)
        sub(/"$/, "", value)
        name = value
        next
      }
      in_package && /^version = "/ {
        value = $0
        sub(/^version = "/, "", value)
        sub(/"$/, "", value)
        version = value
        next
      }
      in_package && /^source = "/ {
        value = $0
        sub(/^source = "/, "", value)
        sub(/"$/, "", value)
        source = value
        next
      }
      in_package && /^checksum = "/ {
        value = $0
        sub(/^checksum = "/, "", value)
        sub(/"$/, "", value)
        checksum = value
        next
      }
      END { emit() }
    ' "$source_root/Cargo.lock")
    record_count=$(printf '%s\n' "$lock_records" \
      | awk 'NF { count++ } END { print count + 0 }')
    [[ "$record_count" == 1 ]] || {
        echo "Cargo.lock must contain exactly one mlx-native package" >&2
        return 1
    }
    IFS=$'\t' read -r lock_version lock_source lock_checksum <<<"$lock_records"
    [[ "$lock_version" == "$manifest_version" ]] || {
        echo "mlx-native manifest and lock versions differ" >&2
        return 1
    }
    [[ "$lock_source" == \
      'registry+https://github.com/rust-lang/crates.io-index' ]] || {
        echo "mlx-native must resolve from the crates.io registry" >&2
        return 1
    }
    [[ "$lock_checksum" =~ ^[0-9a-f]{64}$ ]] || {
        echo "mlx-native registry checksum is missing or malformed" >&2
        return 1
    }
    printf '%s\t%s\t%s\n' \
      "$lock_version" "$lock_source" "$lock_checksum"
}

qwen38_validate_matrix_artifacts() {
    local receipt_path=$1
    local result_path=$2
    local format expected_file expected_bytes expected_sha256 expected_file_type
    local actual_file actual_bytes actual_sha256 actual_file_type

    for format in $(qwen38_artifact_formats); do
        IFS=$'\t' read -r _format expected_file expected_bytes \
            expected_sha256 expected_file_type \
            <<<"$(qwen38_artifact_record "$format")" || return 1
        actual_file=$(jq -er --arg format "$format" \
            "$result_path | select(.format == \$format) | .file" \
            "$receipt_path") || return 1
        actual_bytes=$(jq -er --arg format "$format" \
            "$result_path | select(.format == \$format) | .bytes" \
            "$receipt_path") || return 1
        actual_sha256=$(jq -er --arg format "$format" \
            "$result_path | select(.format == \$format) | .sha256" \
            "$receipt_path") || return 1
        actual_file_type=$(jq -er --arg format "$format" \
            "$result_path | select(.format == \$format) | .file_type" \
            "$receipt_path") || return 1
        [[ "$actual_file" == "$expected_file" ]] || {
            echo "matrix artifact file mismatch for $format" >&2
            return 1
        }
        qwen38_validate_artifact_identity "$format" "$actual_sha256" \
            "$actual_bytes" "$actual_file_type" || return 1
    done
}

qwen38_validate_four_position_matrix_receipt() {
    local receipt_path=$1
    [[ -f "$receipt_path" ]] || return 1
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen38-four-position-artifact-matrix"
      and (.source_commit | test("^[0-9a-f]{40}$"))
      and .dependency.name == "mlx-native"
      and (.dependency.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
      and .dependency.source ==
        "registry+https://github.com/rust-lang/crates.io-index"
      and (.dependency.checksum | test("^[0-9a-f]{64}$"))
      and (.evidence.runner_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.artifact_contract_sha256 | test("^[0-9a-f]{64}$"))
      and .repository == $repository and .revision == $revision
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and (.results | map(.format)) == .formats
      and all(.results[];
        .schema == 2 and .verdict == "pass"
        and .proof.native_storage_without_substitution == true
        and .proof.four_position_matches_scalar == true
        and .proof.eight_token_post_batch_handoff_matches_scalar == true
        and .proof.stable_boundary_compound_matches_split == true
        and .proof.rectangular_state_and_continuation_match_scalar == true
        and (.proof.decode_route | del(.q5k_canonical_q4x4)) ==
          {mlx_disp_bucket:1,mvn:true,mv_ext:false}
        and .proof.decode_route.q5k_canonical_q4x4 == true
        and (.log.path | test("^(bf16|q4_k_m|q5_k_m|q6_k|q8_0)\\.log$"))
        and (.log.sha256 | test("^[0-9a-f]{64}$")))
      and ([.results[] | {
          key:.format,
          value:.proof.exact_mvn_width_four_qtype
        }] | from_entries) == {
          BF16:null,
          Q4_K_M:"Q4_K",
          Q5_K_M:"Q5_K",
          Q6_K:"Q6_K",
          Q8_0:null
        }
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format, file:(.artifact.path | split("/")[-2:] | join("/")), bytes:.artifact.bytes, sha256:.artifact.sha256, file_type:.artifact.gguf_file_type}'
}

qwen38_validate_four_position_matrix_seal() {
    local receipt_path=$1
    local source_root=${2:-}
    local receipt_dir receipt_name evidence_path result_path
    local matrix_sha evidence_sha expected_paths actual_paths
    local expected_entries actual_entries
    local format slug cell_path log_path expected_cell actual_cell
    local source_identity receipt_identity

    [[ -f "$receipt_path" && -r "$receipt_path" && ! -L "$receipt_path" ]] \
      || return 1
    receipt_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    receipt_name=$(basename "$receipt_path")
    [[ "$receipt_name" == matrix.json ]] || {
        echo "four-position matrix receipt must be named matrix.json" >&2
        return 1
    }
    evidence_path="$receipt_dir/evidence.sha256"
    result_path="$receipt_dir/result.sha256"
    [[ -f "$evidence_path" && ! -L "$evidence_path" \
      && -f "$result_path" && ! -L "$result_path" ]] || {
        echo "four-position matrix evidence seal is incomplete" >&2
        return 1
    }
    qwen38_validate_four_position_matrix_receipt "$receipt_path" || return 1
    if [[ -n "$source_root" ]]; then
        source_identity=$(qwen38_mlx_native_registry_identity "$source_root") \
          || return 1
        receipt_identity=$(jq -er '
          [.dependency.version,.dependency.source,.dependency.checksum] | @tsv
        ' "$receipt_path") || return 1
        [[ "$receipt_identity" == "$source_identity" ]] || {
            echo "four-position receipt dependency identity differs from source" >&2
            return 1
        }
    fi
    qwen38_validate_evidence_manifest_paths "$evidence_path" || return 1
    jq -e \
        --arg runner "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_four_position_artifact_matrix.sh" \
          | awk '{print $1}')" \
        --arg contract "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_artifact_contract.sh" \
          | awk '{print $1}')" '
      .evidence.runner_sha256 == $runner
      and .evidence.artifact_contract_sha256 == $contract
    ' "$receipt_path" >/dev/null || return 1

    expected_paths='bf16.json
bf16.log
q4_k_m.json
q4_k_m.log
q5_k_m.json
q5_k_m.log
q6_k.json
q6_k.log
q8_0.json
q8_0.log'
    actual_paths=$(awk '{ print substr($0, 67) }' "$evidence_path" | sort)
    [[ "$actual_paths" == "$expected_paths" ]] || {
        echo "four-position evidence manifest is incomplete or contains extras" >&2
        return 1
    }
    expected_entries='bf16.json
bf16.log
evidence.sha256
matrix.json
q4_k_m.json
q4_k_m.log
q5_k_m.json
q5_k_m.log
q6_k.json
q6_k.log
q8_0.json
q8_0.log
result.sha256'
    actual_entries=$(cd "$receipt_dir" \
      && find . -mindepth 1 -maxdepth 1 -print | sed 's#^./##' | sort) \
      || return 1
    [[ "$actual_entries" == "$expected_entries" ]] || {
        echo "four-position evidence directory is incomplete or contains extras" >&2
        return 1
    }
    for format in $(qwen38_artifact_formats); do
        slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
        cell_path="$receipt_dir/$slug.json"
        log_path="$receipt_dir/$slug.log"
        [[ -f "$cell_path" && ! -L "$cell_path" \
          && -f "$log_path" && ! -L "$log_path" ]] || return 1
        expected_cell=$(jq -Sce --arg format "$format" \
          '.results[] | select(.format == $format)' "$receipt_path") || return 1
        actual_cell=$(jq -Sce . "$cell_path") || return 1
        [[ "$actual_cell" == "$expected_cell" ]] || return 1
        [[ "$(jq -er '.log.path' "$cell_path")" == "$slug.log" \
          && "$(jq -er '.log.sha256' "$cell_path")" == \
            "$(shasum -a 256 "$log_path" | awk '{print $1}')" ]] || return 1
    done
    grep -Fq \
      '[QWEN38_DENSE_ROUTE] qtype=Q5_K kernel=kernel_mul_mv_ext_q5_K_f32_r1_4' \
      "$receipt_dir/q5_k_m.log" || {
        echo "four-position Q5_K evidence lacks the canonical q4x4 dispatch canary" >&2
        return 1
    }
    [[ "$(awk 'END { print NR }' "$result_path")" == 2 ]] || return 1
    matrix_sha=$(shasum -a 256 "$receipt_path" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence_path" | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$result_path")" == "$matrix_sha  matrix.json" \
      && "$(sed -n '2p' "$result_path")" == \
        "$evidence_sha  evidence.sha256" ]] || return 1
    (cd "$receipt_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
      && shasum -a 256 -c result.sha256 >/dev/null)
}

# Copy exactly one already-validated four-position seal. Physical-width
# evidence joins this immutable route proof instead of inferring dispatch from
# a requested environment setting or duplicating the full-model test.
qwen38_copy_four_position_matrix_seal() {
    local receipt_path=$1 destination=$2 source_root=${3:-}
    local source_dir line relative source target
    qwen38_validate_four_position_matrix_seal "$receipt_path" "$source_root" \
      || return 1
    [[ ! -L "$destination" && ( ! -e "$destination" \
      || -z "$(find "$destination" -mindepth 1 -print -quit 2>/dev/null)" ) ]] \
      || return 1
    source_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    mkdir -p "$destination"
    while IFS= read -r line; do
        relative=${line:66}
        source="$source_dir/$relative"
        target="$destination/$relative"
        [[ -f "$source" && ! -L "$source" ]] || return 1
        mkdir -p "$(dirname "$target")"
        cp "$source" "$target"
    done <"$source_dir/evidence.sha256"
    cp "$source_dir/evidence.sha256" "$destination/evidence.sha256"
    cp "$source_dir/result.sha256" "$destination/result.sha256"
    cp "$receipt_path" "$destination/matrix.json"
    qwen38_validate_four_position_matrix_seal \
      "$destination/matrix.json" "$source_root"
}

qwen38_validate_physical_matrix_receipt() {
    local receipt_path=$1 receipt_dir four_position_receipt expected_four_sha
    [[ -f "$receipt_path" ]] || return 1
    receipt_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    four_position_receipt="$receipt_dir/four-position/matrix.json"
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
        --argjson max_tokens "$QWEN38_PHYSICAL_MAX_TOKENS" \
        --argjson kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
        --argjson decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
        --argjson decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" '
      .schema == 2 and .verdict == "pass"
      and .gate == "qwen38-artifact-physical-width-matrix"
      and .repository == $repository and .revision == $revision
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and .widths == [1,2,4,8,16]
      and .workload == {max_tokens:$max_tokens,
        kv_cache_budget_bytes:$kv_budget,
        routing:{decode_mvn:$decode_mvn,decode_mv_ext:$decode_mv_ext,
          q5k_canonical_q4x4:true}}
      and .route_proof.q5k == {qtype:"Q5_K",width:4,
        kernel:"kernel_mul_mv_ext_q5_K_f32_r1_4",actual_dispatch:true}
      and (.route_proof.four_position_matrix_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.matrix_runner_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.gate_runner_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.physical_contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.artifact_contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.results | map(.model.format)) == .formats
      and ([.results[].binary.sha256] | unique | length) == 1
      and (.results[0].binary.sha256 | test("^[0-9a-f]{64}$"))
      and all(.results[];
        .schema == 2 and .verdict == "pass"
        and .model.repository == $repository and .model.revision == $revision
        and .workload.widths == [1,2,4,8,16]
        and .workload.max_tokens == $max_tokens
        and .workload.server_restart_per_width == true
        and .workload.kv_cache_budget_bytes == $kv_budget
        and .workload.routing ==
          {decode_mvn:$decode_mvn,decode_mv_ext:$decode_mv_ext,
            q5k_canonical_q4x4:true}
        and .workload.exact_scalar_replay_per_lane == true
        and (.results | map(.width)) == [1,2,4,8,16]
        and all(.results[];
          .width as $width
          | .schema == 2 and .verdict == "pass"
          and .runtime == {routing:{q5k_canonical_q4x4:true},
            frozen_qwen_policy_log_count:1}
          and .request.exact_scalar_replay_per_lane == true
          and .metrics.scheduler_max_width == .width
          and .metrics.target_body_max_width == .width
          and .metrics.target_head_max_width == .width
          and .metrics.target_forwards_delta > 0
          and .metrics.target_body_rows_delta > 0
          and .metrics.target_head_rows_delta > 0
          and .metrics.command_buffer_submissions_delta > 0
          and (.clients | length) == .width
          and ([.clients[].lane] | sort) == [range(1; $width + 1)]
          and all(.clients[]; .scalar_parity == true)))
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_four_position_matrix_seal "$four_position_receipt" \
      || return 1
    expected_four_sha=$(jq -er '.route_proof.four_position_matrix_sha256' \
      "$receipt_path") || return 1
    [[ "$expected_four_sha" == \
      "$(shasum -a 256 "$four_position_receipt" | awk '{print $1}')" ]] \
      || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format:.model.format, file:.model.file, bytes:.model.bytes, sha256:.model.sha256, file_type:.model.gguf_file_type}'
}

qwen38_validate_evidence_manifest_paths() {
    local manifest=$1
    awk '
      {
        digest = substr($0, 1, 64)
        separator = substr($0, 65, 2)
        path = substr($0, 67)
        if (length(digest) != 64 || digest !~ /^[0-9a-f]+$/ ||
            separator != "  " || length(path) == 0 || path ~ /^\// ||
            path ~ /(^|\/)\.\.?($|\/)/ || path == "matrix.json" ||
            path == "summary.json" || path == "evidence.sha256" ||
            path == "result.sha256" || seen[path]++) {
          invalid++
        }
        rows++
      }
      END { exit !(rows > 0 && invalid == 0) }
    ' "$manifest"
}

qwen38_validate_physical_matrix_seal() {
    local receipt_path=$1
    local receipt_dir receipt_name evidence_path result_path
    local matrix_sha evidence_sha format slug width log_path

    [[ -f "$receipt_path" && -r "$receipt_path" && ! -L "$receipt_path" ]] \
      || return 1
    receipt_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    receipt_name=$(basename "$receipt_path")
    [[ "$receipt_name" == matrix.json ]] || {
        echo "physical matrix receipt must be named matrix.json" >&2
        return 1
    }
    evidence_path="$receipt_dir/evidence.sha256"
    result_path="$receipt_dir/result.sha256"
    [[ -f "$evidence_path" && ! -L "$evidence_path" \
      && -f "$result_path" && ! -L "$result_path" ]] || {
        echo "physical matrix evidence seal is incomplete" >&2
        return 1
    }
    qwen38_validate_physical_matrix_receipt "$receipt_path" || return 1
    qwen38_validate_evidence_manifest_paths "$evidence_path" || return 1
    for format in $(qwen38_artifact_formats); do
        slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
        for width in 1 2 4 8 16; do
            log_path="$receipt_dir/$slug/width-$width/server.log"
            [[ -f "$log_path" && ! -L "$log_path" ]] || return 1
            EXPECTED_Q5K_POLICY=true perl -ne '
              if (/frozen Qwen GGML routing policy/) {
                $seen++;
                $q5=$1 if /dense_q5k_canonical_q4x4=(true|false)/;
              }
              END {exit 1 unless $seen == 1 && $q5 eq $ENV{EXPECTED_Q5K_POLICY}}
            ' "$log_path" || return 1
        done
    done
    jq -e \
        --arg matrix "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_physical_multislot_matrix.sh" \
          | awk '{print $1}')" \
        --arg gate "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_physical_multislot_gate.sh" \
          | awk '{print $1}')" \
        --arg physical_contract "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_physical_multislot_contract.sh" \
          | awk '{print $1}')" \
        --arg artifact_contract "$(shasum -a 256 \
          "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_artifact_contract.sh" \
          | awk '{print $1}')" '
      .evidence.matrix_runner_sha256 == $matrix
      and .evidence.gate_runner_sha256 == $gate
      and .evidence.physical_contract_sha256 == $physical_contract
      and .evidence.artifact_contract_sha256 == $artifact_contract
    ' "$receipt_path" >/dev/null || return 1
    [[ "$(awk 'END { print NR }' "$result_path")" == 2 ]] || return 1
    matrix_sha=$(shasum -a 256 "$receipt_path" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence_path" | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$result_path")" == "$matrix_sha  matrix.json"
      && "$(sed -n '2p' "$result_path")" == "$evidence_sha  evidence.sha256" ]] \
      || return 1
    (cd "$receipt_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
      && shasum -a 256 -c result.sha256 >/dev/null)
}

# Copy exactly the files named by a validated physical-matrix seal. This keeps
# the matched result self-contained without recursively copying an arbitrary
# source directory or trusting manifest path traversal/symlinks.
qwen38_copy_physical_matrix_seal() {
    local receipt_path=$1 destination=$2 source_dir line relative source target
    qwen38_validate_physical_matrix_seal "$receipt_path" || return 1
    [[ ! -L "$destination" && ( ! -e "$destination" \
      || -z "$(find "$destination" -mindepth 1 -print -quit 2>/dev/null)" ) ]] \
      || return 1
    source_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    mkdir -p "$destination"
    while IFS= read -r line; do
        relative=${line:66}
        source="$source_dir/$relative"
        target="$destination/$relative"
        [[ -f "$source" && ! -L "$source" ]] || return 1
        mkdir -p "$(dirname "$target")"
        cp "$source" "$target"
    done <"$source_dir/evidence.sha256"
    cp "$source_dir/evidence.sha256" "$destination/evidence.sha256"
    cp "$source_dir/result.sha256" "$destination/result.sha256"
    cp "$receipt_path" "$destination/matrix.json"
    qwen38_validate_physical_matrix_seal "$destination/matrix.json"
}

qwen38_validate_matched_peer_matrix_receipt() {
    local receipt_path=$1
    local pinned_peer_commit
    [[ -f "$receipt_path" ]] || return 1
    pinned_peer_commit=$(qwen38_pinned_peer_commit) || return 1
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
        --arg peer_commit "$pinned_peer_commit" \
        --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
        --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
        --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
        --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
        --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
        --argjson decode_mvn 1 \
        --argjson decode_mv_ext 0 \
        --argjson q5k "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
        --argjson hf2q_kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
        --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" '
      .schema == 2 and .verdict == "pass"
      and .gate == "qwen38-matched-peer-artifact-matrix"
      and .repository == $repository and .revision == $revision
      and .pinned_peer_commit == $peer_commit
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and (.results | map(.model.format)) == .formats
      and ([.results[].hf2q.commit] | unique | length) == 1
      and (.results[0].hf2q.commit | test("^[0-9a-f]{40}$"))
      and ([.results[].hf2q.binary_sha256] | unique | length) == 1
      and (.results[0].hf2q.binary_sha256 | test("^[0-9a-f]{64}$"))
      and .hf2q_effective_routing_policy == {
        dense_decode_mvn:$decode_mvn,
        dense_decode_mv_ext:$decode_mv_ext,
        dense_q5k_canonical_q4x4:$q5k}
      and ([.results[].hf2q.effective_routing_policy] | unique) ==
        [.hf2q_effective_routing_policy]
      and all(.results[];
        .schema == 5 and .verdict == "pass"
        and .reference.commit == $peer_commit
        and .model.repository == $repository and .model.revision == $revision
        and .quality.code.evaluator_tests_passed == true
        and .quality.repeat.exact_expected_content == true
        and .stability.stable == true
        and .launch_settings.schema == 2
        and .launch_settings.hf2q.dense_decode_mvn == $decode_mvn
        and .launch_settings.hf2q.dense_decode_mv_ext == $decode_mv_ext
        and .launch_settings.hf2q.dense_q5k_canonical_q4x4 == $q5k
        and .hf2q.effective_routing_policy ==
          (.launch_settings.hf2q
           | {dense_decode_mvn,dense_decode_mv_ext,
               dense_q5k_canonical_q4x4})
        and .launch_settings.hf2q.speculation == $hf2q_speculation
        and .launch_settings.reference.speculation == $reference_speculation
        and .launch_settings.hf2q.kv_cache == $hf2q_kv
        and .launch_settings.reference.kv_cache_k == $reference_k
        and .launch_settings.reference.kv_cache_v == $reference_v
        and .launch_settings.hf2q.kv_cache_budget_bytes == $hf2q_kv_budget
        and .launch_settings.hf2q.context_tokens_per_slot == $context_tokens
        and .launch_settings.reference.context_tokens_total == $context_tokens
        and .hf2q.speculation == .launch_settings.hf2q.speculation
        and .reference.speculation == .launch_settings.reference.speculation
        and .acceptance.minimum_hf2q_ratio >= 1
        and .code.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio
        and .repeat.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio)
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format:.model.format, file:.model.file, bytes:.model.bytes, sha256:.model.sha256, file_type:.model.gguf_file_type}'
}

qwen38_validate_matched_physical_matrix_receipt() {
    local receipt_path=$1 pinned_reference_commit receipt_dir physical_path
    local child_path summary_sha evidence_sha result_sha child_index=0
    local format width expected_proof actual_proof
    [[ -f "$receipt_path" ]] || return 1
    pinned_reference_commit=$(qwen38_pinned_peer_commit) || return 1
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
        --arg reference_commit "$pinned_reference_commit" \
        --argjson kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
        --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
        --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
        --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
        --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
        --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
        --argjson decode_mvn 1 \
        --argjson decode_mv_ext 0 \
        --argjson q5k "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
        --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" \
        --argjson max_launch_skew \
          "$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS" '
      def valid_wave:
        .schema == 1 and .proof_pass == true
        and (.group == "code" or .group == "repeat")
        and .policy == (if .engine == "hf2q" then $hf2q_speculation
          else $reference_speculation end)
        and (.proposals | type == "number")
        and .proposals == (.proposals | floor) and .proposals > 0
        and (.drafted_tokens | type == "number")
        and .drafted_tokens == (.drafted_tokens | floor) and .drafted_tokens > 0
        and (.accepted_tokens | type == "number")
        and .accepted_tokens == (.accepted_tokens | floor)
        and .accepted_tokens >= 0
        and .accepted_tokens <= .drafted_tokens
        and (.cost_disabled_generations | type == "number")
        and .cost_disabled_generations == (.cost_disabled_generations | floor)
        and .cost_disabled_generations >= 0
        and (.measured_round_seconds | type == "number")
        and .measured_round_seconds >= 0
        and (.equivalent_ordinary_seconds | type == "number")
        and .equivalent_ordinary_seconds >= 0
        and (if .engine == "reference" then
            .proof_mode == "accepted-proposals"
            and .accepted_tokens > 0 and .disable_reason == null
          elif .engine == "hf2q" and .proof_mode == "accepted-proposals" then
            .accepted_tokens > 0 and .disable_reason == null
          elif .engine == "hf2q" and .proof_mode == "measured-cost-disabled" then
            .accepted_tokens == 0 and .cost_disabled_generations > 0
            and .measured_round_seconds > 0
            and .equivalent_ordinary_seconds > 0
            and .disable_reason == "measured_cost_unprofitable"
          else false end);
      . as $matrix
      | .schema == 2 and .verdict == "pass"
      and .gate == "qwen38-matched-physical-artifact-matrix"
      and .repository == $repository and .revision == $revision
      and .pinned_reference_commit == $reference_commit
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and .widths == [1,2,4,8,16]
      and .workload == {
        speculation:{hf2q:$hf2q_speculation,reference:$reference_speculation},
        cache_settings:{
          hf2q:{format:$hf2q_kv,budget_bytes:$kv_budget,
            context_tokens_per_slot:$context_tokens},
          reference:{k_format:$reference_k,v_format:$reference_v,
            context_tokens_total:$context_tokens}}}
      and .hf2q_effective_routing_policy == {
        dense_decode_mvn:$decode_mvn,
        dense_decode_mv_ext:$decode_mv_ext,
        dense_q5k_canonical_q4x4:$q5k}
      and .acceptance == {maximum_launch_skew_seconds:$max_launch_skew}
      and (.physical_matrix.sha256 | test("^[0-9a-f]{64}$"))
      and .physical_matrix.gate == "qwen38-artifact-physical-width-matrix"
      and .physical_matrix.seal_validated == true
      and .physical_matrix.self_contained_path == "physical-proof/matrix.json"
      and (.physical_matrix.binary_sha256 | test("^[0-9a-f]{64}$"))
      and .evidence.child_results_sealed == true
      and (.evidence.script_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.artifact_contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.children | length) == 5
      and ([.evidence.children[].format]) == .formats
      and ([.evidence.children[].path]) ==
        ["artifacts/bf16","artifacts/q4_k_m","artifacts/q5_k_m",
         "artifacts/q6_k","artifacts/q8_0"]
      and all(.evidence.children[];
        (.summary_sha256 | test("^[0-9a-f]{64}$"))
        and (.evidence_manifest_sha256 | test("^[0-9a-f]{64}$"))
        and (.result_seal_sha256 | test("^[0-9a-f]{64}$")))
      and (.results | map(.model.format)) == .formats
      and ([.results[].hf2q.commit] | unique | length) == 1
      and (.results[0].hf2q.commit | test("^[0-9a-f]{40}$"))
      and ([.results[].hf2q.binary_sha256] | unique) ==
        [.physical_matrix.binary_sha256]
      and ([.results[].reference.commit] | unique) == [$reference_commit]
      and ([.results[].reference.binary_sha256] | unique | length) == 1
      and (.results[0].reference.binary_sha256 | test("^[0-9a-f]{64}$"))
      and ([.results[].physical_matrix_sha256] | unique) ==
        [.physical_matrix.sha256]
      and ([.results[].hf2q.effective_routing_policy] | unique) ==
        [.hf2q_effective_routing_policy]
      and all(.results[]; . as $artifact
        | .schema == 2 and .verdict == "pass"
        and .gate == "qwen38-matched-physical-abba"
        and .reference.commit == $reference_commit
        and .model.repository == $repository and .model.revision == $revision
        and .workload.widths == [1,2,4,8,16]
        and .workload.trial_order == ["hf2q","reference","reference","hf2q"]
        and .workload.speculation ==
          {hf2q:$hf2q_speculation,reference:$reference_speculation}
        and .workload.cache_settings == {
          hf2q:{format:$hf2q_kv,budget_bytes:$kv_budget,
            context_tokens_per_slot:$context_tokens},
          reference:{k_format:$reference_k,v_format:$reference_v,
            context_tokens_total:$context_tokens}}
        and .workload.scalar_replay_per_lane == true
        and .workload.reference_parallelism_matches_width == true
        and .hf2q.effective_routing_policy ==
          $matrix.hf2q_effective_routing_policy
        and .acceptance.minimum_hf2q_ratio >= 1
        and .acceptance.maximum_launch_skew_seconds == $max_launch_skew
        and (.results | map(.width)) == [1,2,4,8,16]
        and all(.results[]; . as $cell
          | .schema == 2 and .verdict == "pass"
          and .hf2q_effective_routing_policy ==
            $artifact.hf2q.effective_routing_policy
          and .samples.hf2q == 2 and .samples.reference == 2
          and .scalar_replay.hf2q == true and .scalar_replay.reference == true
          and .physical_proof.width == .width
          and .physical_proof.mode == "ordinary-target-speculation-off"
          and .physical_proof.seal_validated == true
          and .physical_proof.scheduler_max_width == .width
          and .physical_proof.target_body_max_width == .width
          and .physical_proof.target_head_max_width == .width
          and .physical_proof.command_buffer_submissions_delta > 0
          and (.physical_proof.clients | length) == .width
          and ([.physical_proof.clients[].lane] | sort) ==
            [range(1; $cell.width + 1)]
          and all(.physical_proof.clients[]; .scalar_parity == true)
          and .speculation.hf2q_policy == $hf2q_speculation
          and .speculation.reference_policy == $reference_speculation
          and ([.speculation.waves[] | [.trial,.engine,.group]]) ==
            [[1,"hf2q","code"],[1,"hf2q","repeat"],
             [2,"reference","code"],[2,"reference","repeat"],
             [3,"reference","code"],[3,"reference","repeat"],
             [4,"hf2q","code"],[4,"hf2q","repeat"]]
          and all(.speculation.waves[]; valid_wave)
          and .acceptance.minimum_hf2q_ratio >= 1
          and .code.width == .width and .code.group == "code"
          and .code.quality_pass == true
          and .code.measurement_consistent == true
          and .code.api_concurrency_pass == true
          and .code.token_accounting.pass == true
          and .code.token_accounting.cross_engine_prompt_equality_required == true
          and .code.token_accounting.raw_api_completion_equality_required_within_engine == true
          and .code.token_accounting.cross_engine_completion_equality_required == false
          and .code.token_accounting.cross_engine_semantic_completion_equality_required == false
          and .code.stability.stable == true
          and .code.stability.observed_band_dominance == true
          and .code.hf2q_over_reference_comparison_rate >=
            .acceptance.minimum_hf2q_ratio
          and .repeat.width == .width and .repeat.group == "repeat"
          and .repeat.quality_pass == true
          and .repeat.measurement_consistent == true
          and .repeat.api_concurrency_pass == true
          and .repeat.token_accounting.pass == true
          and .repeat.token_accounting.cross_engine_prompt_equality_required == true
          and .repeat.token_accounting.raw_api_completion_equality_required_within_engine == true
          and .repeat.token_accounting.cross_engine_completion_equality_required == false
          and .repeat.token_accounting.cross_engine_semantic_completion_equality_required == true
          and .repeat.token_accounting.semantic_tokenization_sha256
            == $artifact.workload.repeat_semantic_tokenization.receipt_sha256
          and .repeat.stability.stable == true
          and .repeat.stability.observed_band_dominance == true
          and .repeat.hf2q_over_reference_comparison_rate >=
            .acceptance.minimum_hf2q_ratio
          and .repeat.reference_over_hf2q_p95_wall >=
            .acceptance.minimum_hf2q_ratio
          and .repeat.semantic_ttft.required == true
          and .repeat.semantic_ttft.stable == true
          and .repeat.semantic_ttft.observed_band_dominance == true
          and .repeat.semantic_ttft.reference_over_hf2q_p95 >=
            .acceptance.minimum_hf2q_ratio))
    ' "$receipt_path" >/dev/null || return 1

    receipt_dir=$(cd "$(dirname "$receipt_path")" && pwd) || return 1
    physical_path="$receipt_dir/physical-proof/matrix.json"
    qwen38_validate_physical_matrix_seal "$physical_path" || return 1
    [[ "$(shasum -a 256 "$physical_path" | awk '{print $1}')" == \
      "$(jq -er '.physical_matrix.sha256' "$receipt_path")" ]] || return 1
    for format in $(qwen38_artifact_formats); do
        for width in 1 2 4 8 16; do
            expected_proof=$(jq -Sce --arg format "$format" \
              --argjson width "$width" '
              .results[] | select(.model.format == $format)
              | .results[] | select(.width == $width)
              | {width,mode:"ordinary-target-speculation-off",seal_validated:true,
                  scheduler_max_width:.metrics.scheduler_max_width,
                  target_body_max_width:.metrics.target_body_max_width,
                  target_head_max_width:.metrics.target_head_max_width,
                  command_buffers_created_delta:
                    .metrics.command_buffers_created_delta,
                  command_buffer_submissions_delta:
                    .metrics.command_buffer_submissions_delta,
                  clients:[.clients[] | {lane,scalar_parity}]}
            ' "$physical_path") || return 1
            actual_proof=$(jq -Sce --arg format "$format" \
              --argjson width "$width" '
              .results[] | select(.model.format == $format)
              | .results[] | select(.width == $width) | .physical_proof
            ' "$receipt_path") || return 1
            [[ "$actual_proof" == "$expected_proof" ]] || return 1
        done
    done
    while IFS=$'\t' read -r child_path summary_sha evidence_sha result_sha; do
        [[ "$child_path" =~ ^artifacts/(bf16|q4_k_m|q5_k_m|q6_k|q8_0)$ ]] \
          || return 1
        qwen38_validate_evidence_manifest_paths \
          "$receipt_dir/$child_path/evidence.sha256" || return 1
        [[ -f "$receipt_dir/$child_path/summary.json" \
          && ! -L "$receipt_dir/$child_path/summary.json" \
          && ! -L "$receipt_dir/$child_path/evidence.sha256" \
          && ! -L "$receipt_dir/$child_path/result.sha256" \
          && "$(awk 'END {print NR}' \
            "$receipt_dir/$child_path/result.sha256")" == 2 \
          && "$(sed -n '1p' "$receipt_dir/$child_path/result.sha256")" == \
            "$summary_sha  summary.json" \
          && "$(sed -n '2p' "$receipt_dir/$child_path/result.sha256")" == \
            "$evidence_sha  evidence.sha256" ]] || return 1
        (cd "$receipt_dir/$child_path" \
          && shasum -a 256 -c evidence.sha256 >/dev/null \
          && shasum -a 256 -c result.sha256 >/dev/null) || return 1
        [[ "$(shasum -a 256 "$receipt_dir/$child_path/summary.json" \
            | awk '{print $1}')" == "$summary_sha" \
          && "$(shasum -a 256 "$receipt_dir/$child_path/evidence.sha256" \
            | awk '{print $1}')" == "$evidence_sha" \
          && "$(shasum -a 256 "$receipt_dir/$child_path/result.sha256" \
            | awk '{print $1}')" == "$result_sha" ]] || return 1
        [[ "$(jq -S -c . "$receipt_dir/$child_path/summary.json")" == \
          "$(jq -S -c --argjson index "$child_index" \
            '.results[$index]' "$receipt_path")" ]] || return 1
        child_index=$((child_index + 1))
    done < <(jq -r '.evidence.children[]
      | [.path,.summary_sha256,.evidence_manifest_sha256,.result_seal_sha256]
      | @tsv' "$receipt_path")
    ((child_index == 5)) || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
      '.results[] | {format:.model.format, file:.model.file, bytes:.model.bytes, sha256:.model.sha256, file_type:.model.gguf_file_type}'
}
