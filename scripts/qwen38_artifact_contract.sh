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
QWEN38_ARTIFACT_CONTRACT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly QWEN38_ARTIFACT_CONTRACT_ROOT_DIR

qwen38_artifact_formats() {
    printf '%s\n' BF16 Q4_K_M Q5_K_M Q6_K Q8_0
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

qwen38_pinned_peer_commit() {
    local pin_path="$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/data/llama_cpp_pin.txt"
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
      and .repository == $repository and .revision == $revision
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and (.results | map(.format)) == .formats
      and all(.results[];
        .verdict == "pass"
        and .proof.native_storage_without_substitution == true
        and .proof.four_position_matches_scalar == true
        and .proof.eight_token_post_batch_handoff_matches_scalar == true
        and (.log.sha256 | test("^[0-9a-f]{64}$")))
      and (.results | map(select(.format == "Q4_K_M"))[0]
        .proof.q4_mv_ext_width_four == true)
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format, file:(.artifact.path | split("/")[-2:] | join("/")), bytes:.artifact.bytes, sha256:.artifact.sha256, file_type:.artifact.gguf_file_type}'
}

qwen38_validate_physical_matrix_receipt() {
    local receipt_path=$1
    [[ -f "$receipt_path" ]] || return 1
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen38-artifact-physical-width-matrix"
      and .repository == $repository and .revision == $revision
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and .widths == [1,2,4,8,16]
      and (.results | map(.model.format)) == .formats
      and ([.results[].binary.sha256] | unique | length) == 1
      and (.results[0].binary.sha256 | test("^[0-9a-f]{64}$"))
      and all(.results[];
        .schema == 1 and .verdict == "pass"
        and .model.repository == $repository and .model.revision == $revision
        and .workload.widths == [1,2,4,8,16]
        and .workload.exact_scalar_replay_per_lane == true
        and (.results | map(.width)) == [1,2,4,8,16]
        and all(.results[];
          .verdict == "pass"
          and .request.exact_scalar_replay_per_lane == true
          and .metrics.scheduler_max_width == .width
          and .metrics.target_body_max_width == .width
          and .metrics.target_head_max_width == .width
          and .metrics.target_forwards_delta > 0
          and .metrics.target_body_rows_delta > 0
          and .metrics.target_head_rows_delta > 0
          and .metrics.command_buffer_submissions_delta > 0
          and (.clients | length) == .width
          and all(.clients[]; .scalar_parity == true)))
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format:.model.format, file:.model.file, bytes:.model.bytes, sha256:.model.sha256, file_type:.model.gguf_file_type}'
}

qwen38_validate_matched_peer_matrix_receipt() {
    local receipt_path=$1
    local pinned_peer_commit
    [[ -f "$receipt_path" ]] || return 1
    pinned_peer_commit=$(qwen38_pinned_peer_commit) || return 1
    jq -e \
        --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
        --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
        --arg peer_commit "$pinned_peer_commit" '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen38-matched-peer-artifact-matrix"
      and .repository == $repository and .revision == $revision
      and .pinned_peer_commit == $peer_commit
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and (.results | map(.model.format)) == .formats
      and ([.results[].hf2q.commit] | unique | length) == 1
      and (.results[0].hf2q.commit | test("^[0-9a-f]{40}$"))
      and ([.results[].hf2q.binary_sha256] | unique | length) == 1
      and (.results[0].hf2q.binary_sha256 | test("^[0-9a-f]{64}$"))
      and all(.results[];
        .verdict == "pass"
        and .reference.commit == $peer_commit
        and .model.repository == $repository and .model.revision == $revision
        and .quality.code.evaluator_tests_passed == true
        and .quality.repeat.exact_expected_content == true
        and .stability.stable == true
        and .acceptance.minimum_hf2q_ratio >= 1
        and .code.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio
        and .repeat.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio)
    ' "$receipt_path" >/dev/null || return 1
    qwen38_validate_matrix_artifacts "$receipt_path" \
        '.results[] | {format:.model.format, file:.model.file, bytes:.model.bytes, sha256:.model.sha256, file_type:.model.gguf_file_type}'
}
