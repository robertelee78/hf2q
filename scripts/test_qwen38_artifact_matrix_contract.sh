#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"

fail() {
    echo "$*" >&2
    exit 1
}

expect_failure() {
    local label=$1
    shift
    if "$@" >/dev/null 2>&1; then
        fail "negative artifact-contract fixture passed: $label"
    fi
}

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-matrix-contract.XXXXXX")
cleanup() {
    case "$test_dir" in
        "${TMPDIR:-/tmp}"/hf2q-qwen38-matrix-contract.*)
            rm -rf "$test_dir"
            ;;
        *)
            echo "refusing unsafe test cleanup: $test_dir" >&2
            ;;
    esac
}
trap cleanup EXIT

expected_formats=(BF16 Q4_K_M Q5_K_M Q6_K Q8_0)
expected_files=(
    gguf/qwen38-abliterated-sft-bf16.gguf
    gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf
    gguf/qwen38-abliterated-sft-q5_k_m.gguf
    gguf/qwen38-abliterated-sft-q6_k.gguf
    gguf/qwen38-abliterated-sft-q8_0.gguf
)
expected_bytes=(54657734208 16810714944 19535701568 22431000128 29047084608)
expected_sha256=(
    f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee
    1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a
    4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e
    78f62a87ef851443d4e0c74c4e1eb1dfe73e3bf0ded3cf320ec80f763020ddb3
    53c076e5117be1391e76a9746998fbe2040e6b69a73aa47d1c1b0ca97a8a2c99
)
expected_file_types=(32 15 17 18 7)

actual_formats=$(qwen38_artifact_formats | tr '\n' ' ' | sed 's/ $//')
[[ "$actual_formats" == "${expected_formats[*]}" ]] \
    || fail "artifact format catalog drifted: $actual_formats"

for index in "${!expected_formats[@]}"; do
    format=${expected_formats[$index]}
    record=$(qwen38_artifact_record "$format")
    IFS=$'\t' read -r actual_format file bytes sha256 file_type <<<"$record"
    [[ "$actual_format" == "$format" \
        && "$file" == "${expected_files[$index]}" \
        && "$bytes" == "${expected_bytes[$index]}" \
        && "$sha256" == "${expected_sha256[$index]}" \
        && "$file_type" == "${expected_file_types[$index]}" ]] \
        || fail "artifact record drifted for $format: $record"
    qwen38_validate_artifact_identity \
        "$format" "$sha256" "$bytes" "$file_type"
    expect_failure "$format-wrong-sha" qwen38_validate_artifact_identity \
        "$format" "${sha256%?}0" "$bytes" "$file_type"
    expect_failure "$format-wrong-bytes" qwen38_validate_artifact_identity \
        "$format" "$sha256" "$((bytes + 1))" "$file_type"
    expect_failure "$format-wrong-file-type" qwen38_validate_artifact_identity \
        "$format" "$sha256" "$bytes" "$((file_type + 1))"
done
# shellcheck disable=SC2016
grep -Fq 'cargo test --locked --bin hf2q "$test_name"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not run the owning hf2q test target"
grep -Fq 'test result: ok\. 1 passed; 0 failed; 0 ignored;' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not reject a zero-test cargo pass"

expect_failure unknown-format qwen38_artifact_record Q3_K_M
expect_failure case-folded-format qwen38_artifact_record q5_k_m

pin=$(qwen38_pinned_peer_commit)
[[ "$pin" =~ ^[0-9a-f]{40}$ ]] || fail "pinned peer commit is not exact: $pin"
[[ "$pin" == "3f545beccee69d9975f466ec7e45fd9aacd8ba90" ]] \
    || fail "pinned peer commit drifted from regenerated fixture authority: $pin"
qwen38_validate_pinned_peer_commit "$pin"
expect_failure wrong-peer-commit qwen38_validate_pinned_peer_commit \
    0000000000000000000000000000000000000000

four_cells="$test_dir/four-cells.jsonl"
physical_cells="$test_dir/physical-cells.jsonl"
matched_cells="$test_dir/matched-cells.jsonl"
: >"$four_cells"
: >"$physical_cells"
: >"$matched_cells"
for format in "${expected_formats[@]}"; do
    IFS=$'\t' read -r _format file bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" \
        --arg log_sha256 1111111111111111111111111111111111111111111111111111111111111111 \
        --argjson q4_mv_ext "$([[ "$format" == Q4_K_M ]] && echo true || echo false)" '{
          verdict:"pass",format:$format,
          artifact:{path:("/models/" + $file),sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          proof:{native_storage_without_substitution:true,
            four_position_matches_scalar:true,
            eight_token_post_batch_handoff_matches_scalar:true,
            q4_mv_ext_width_four:$q4_mv_ext},
          log:{sha256:$log_sha256}
        }' >>"$four_cells"
    width_results=$(jq -nc '[1,2,4,8,16] | map(. as $width | {
      verdict:"pass",width:$width,
      request:{exact_scalar_replay_per_lane:true},
      metrics:{scheduler_max_width:$width,target_body_max_width:$width,
        target_head_max_width:$width,target_forwards_delta:1,
        target_body_rows_delta:$width,target_head_rows_delta:$width,
        command_buffer_submissions_delta:1},
      clients:[range(0;$width) | {scalar_parity:true}]
    })')
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" --argjson results "$width_results" '{
          schema:1,verdict:"pass",
          binary:{sha256:("2" * 64)},
          model:{format:$format,file:$file,
            repository:"jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            revision:"0a72776892f98db49381fdf69f4b9982222ec9dc",
            sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          workload:{widths:[1,2,4,8,16],exact_scalar_replay_per_lane:true},
          results:$results
        }' >>"$physical_cells"
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" --arg peer_commit "$pin" '{
          verdict:"pass",hf2q:{commit:("3" * 40),binary_sha256:("4" * 64)},
          reference:{commit:$peer_commit},
          model:{format:$format,file:$file,
            repository:"jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            revision:"0a72776892f98db49381fdf69f4b9982222ec9dc",
            sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          quality:{code:{evaluator_tests_passed:true},
            repeat:{exact_expected_content:true}},
          stability:{stable:true},acceptance:{minimum_hf2q_ratio:1},
          code:{hf2q_over_reference:1.01},repeat:{hf2q_over_reference:1.01}
        }' >>"$matched_cells"
done

jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '{
      schema:1,verdict:"pass",gate:"qwen38-four-position-artifact-matrix",
      source_commit:("5" * 40),repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:.
    }' "$four_cells" >"$test_dir/four.json"
jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '{
      schema:1,verdict:"pass",gate:"qwen38-artifact-physical-width-matrix",
      repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
      widths:[1,2,4,8,16],results:.
    }' "$physical_cells" >"$test_dir/physical.json"
jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg peer_commit "$pin" '{
      schema:1,verdict:"pass",gate:"qwen38-matched-peer-artifact-matrix",
      repository:$repository,revision:$revision,pinned_peer_commit:$peer_commit,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:.
    }' "$matched_cells" >"$test_dir/matched.json"

qwen38_validate_four_position_matrix_receipt "$test_dir/four.json"
qwen38_validate_physical_matrix_receipt "$test_dir/physical.json"
qwen38_validate_matched_peer_matrix_receipt "$test_dir/matched.json"

jq 'del(.results[4])' "$test_dir/four.json" >"$test_dir/four-missing-format.json"
expect_failure four-missing-format qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-missing-format.json"
jq '.results[1].proof.q4_mv_ext_width_four = false' \
    "$test_dir/four.json" >"$test_dir/four-missing-route-proof.json"
expect_failure four-missing-route-proof qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-missing-route-proof.json"
jq 'del(.results[2].results[4])' \
    "$test_dir/physical.json" >"$test_dir/physical-missing-width.json"
expect_failure physical-missing-width qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-missing-width.json"
jq '.results[3].results[3].clients[0].scalar_parity = false' \
    "$test_dir/physical.json" >"$test_dir/physical-scalar-divergence.json"
expect_failure physical-scalar-divergence qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-scalar-divergence.json"
jq '.results[1].results[1].metrics.target_body_max_width = 1' \
    "$test_dir/physical.json" >"$test_dir/physical-false-width-telemetry.json"
expect_failure physical-false-width-telemetry qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-false-width-telemetry.json"
jq '.results[0].model.sha256 = ("0" * 64)' \
    "$test_dir/physical.json" >"$test_dir/physical-wrong-artifact.json"
expect_failure physical-wrong-artifact qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-wrong-artifact.json"
jq '.results[4].verdict = "skip"' \
    "$test_dir/matched.json" >"$test_dir/matched-skipped-cell.json"
expect_failure matched-skipped-cell qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-skipped-cell.json"
jq '.results[0].code.hf2q_over_reference = 0.99' \
    "$test_dir/matched.json" >"$test_dir/matched-slower-cell.json"
expect_failure matched-slower-cell qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-slower-cell.json"
jq '.results[0].acceptance.minimum_hf2q_ratio = 0.5' \
    "$test_dir/matched.json" >"$test_dir/matched-weakened-threshold.json"
expect_failure matched-weakened-threshold qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-weakened-threshold.json"
jq '.pinned_peer_commit = ("0" * 40)' \
    "$test_dir/matched.json" >"$test_dir/matched-wrong-peer.json"
expect_failure matched-wrong-peer qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-wrong-peer.json"

for runner in \
    "$script_dir/qwen38_matched_reference_abba.sh" \
    "$script_dir/qwen38_physical_multislot_gate.sh"; do
    grep -Fq 'qwen38_validate_artifact_identity' "$runner" \
        || fail "runner does not call the shared artifact identity gate: $runner"
done
grep -Fq 'qwen38_validate_pinned_peer_commit' \
    "$script_dir/qwen38_matched_reference_abba.sh" \
    || fail "matched runner does not enforce data/llama_cpp_pin.txt"

for matrix_runner in \
    "$script_dir/qwen38_matched_peer_matrix.sh" \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh"; do
    bash -n "$matrix_runner"
    grep -Fq 'qwen38_artifact_formats' "$matrix_runner" \
        || fail "matrix runner does not enumerate the authoritative catalog: $matrix_runner"
    grep -Fq 'matrix.json.tmp' "$matrix_runner" \
        || fail "matrix runner does not publish summary last: $matrix_runner"
done
grep -Fq 'qwen38_validate_four_position_matrix_receipt "$OUT_DIR/matrix.json.tmp"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not execute its fail-closed receipt validator"
grep -Fq 'qwen38_validate_physical_matrix_receipt "$OUT_DIR/matrix.json.tmp"' \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    || fail "physical matrix does not execute its fail-closed receipt validator"
grep -Fq 'qwen38_validate_matched_peer_matrix_receipt "$OUT_DIR/matrix.json.tmp"' \
    "$script_dir/qwen38_matched_peer_matrix.sh" \
    || fail "matched matrix does not execute its fail-closed receipt validator"

bash -n "$script_dir/qwen38_artifact_contract.sh"
bash -n "$script_dir/qwen38_matched_reference_abba.sh"
bash -n "$script_dir/qwen38_physical_multislot_gate.sh"

echo "Qwen3.8 artifact matrix contract: PASS"
