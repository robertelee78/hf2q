#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen_family_mtp_artifact_contract.sh
source "$script_dir/qwen_family_mtp_artifact_contract.sh"

fail() { echo "$*" >&2; exit 1; }
expect_failure() {
    local label=$1
    shift
    if "$@" >/dev/null 2>&1; then
        fail "negative Qwen family MTP fixture passed: $label"
    fi
}

expected_cells='qwen35-dense
qwen36-dense
qwen36-moe'
[[ "$(qwen_family_mtp_cells)" == "$expected_cells" ]] \
    || fail 'Qwen family MTP cell order drifted'
for cell in $(qwen_family_mtp_cells); do
    IFS=$'\t' read -r actual _repository _revision _file bytes sha _arch file_type \
        _target_layers _total_blocks _variant <<<"$(qwen_family_mtp_record "$cell")"
    [[ "$actual" == "$cell" && "$file_type" == 15 && "$sha" =~ ^[0-9a-f]{64}$ ]] \
        || fail "invalid immutable record for $cell"
    qwen_family_mtp_validate_artifact_identity "$cell" "$bytes" "$sha"
    expect_failure "$cell-wrong-size" qwen_family_mtp_validate_artifact_identity \
        "$cell" "$((bytes + 1))" "$sha"
done

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen-family-mtp-contract.XXXXXX")
cleanup() {
    case "$test_dir/" in
        "${TMPDIR:-/tmp}"/hf2q-qwen-family-mtp-contract.*/*)
            rm -rf -- "$test_dir"
            ;;
        *) echo "refusing unsafe Qwen family contract cleanup: $test_dir" >&2 ;;
    esac
}
trap cleanup EXIT

seal_family_fixture() {
    local root=$1 file
    : >"$root/evidence.sha256"
    for file in qwen35-dense.json qwen35-dense.log \
        qwen36-dense.json qwen36-dense.log qwen36-moe.json qwen36-moe.log; do
        printf '%s  %s\n' "$(shasum -a 256 "$root/$file" | awk '{print $1}')" "$file" \
            >>"$root/evidence.sha256"
    done
    printf '%s  matrix.json\n%s  evidence.sha256\n' \
        "$(shasum -a 256 "$root/matrix.json" | awk '{print $1}')" \
        "$(shasum -a 256 "$root/evidence.sha256" | awk '{print $1}')" \
        >"$root/result.sha256"
}

copy_family_fixture() {
    local source=$1 destination=$2 file
    mkdir -p "$destination"
    for file in matrix.json evidence.sha256 result.sha256 \
        qwen35-dense.json qwen35-dense.log \
        qwen36-dense.json qwen36-dense.log qwen36-moe.json qwen36-moe.log; do
        cp "$source/$file" "$destination/$file"
    done
}

results='[]'
for cell in $(qwen_family_mtp_cells); do
    IFS=$'\t' read -r _cell repository revision file bytes sha arch file_type \
        target_layers total_blocks variant <<<"$(qwen_family_mtp_record "$cell")"
    printf '%s\n' "$cell exact native proof" >"$test_dir/$cell.log"
    entry=$(jq -n --arg cell "$cell" --arg repository "$repository" \
        --arg revision "$revision" --arg file "$file" --arg sha "$sha" \
        --arg arch "$arch" --arg variant "$variant" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" --argjson target_layers "$target_layers" \
        --argjson total_blocks "$total_blocks" --arg log_path "$cell.log" \
        --arg log_sha "$(shasum -a 256 "$test_dir/$cell.log" | awk '{print $1}')" '{
          schema:1,verdict:"pass",cell:$cell,repository:$repository,revision:$revision,
          artifact:{file:$file,sha256:$sha,bytes:$bytes,file_type:$file_type,
            architecture:$arch,variant:$variant,target_layers:$target_layers,
            total_blocks:$total_blocks,mtp_layers:1},
          proof:{metadata_preflight_before_metal:true,
            native_storage_without_substitution:true,
            shared_embedding_and_exact_head_allocation:true,
            mtp_compound_matches_sequential:true,
            target_and_mtp_final_bytes_equal:true,
            quantized_dispatch_observed:true},
          log:{path:$log_path,sha256:$log_sha}}')
    printf '%s\n' "$entry" >"$test_dir/$cell.json"
    results=$(jq -nc --argjson prior "$results" --argjson entry "$entry" '$prior + [$entry]')
done

IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
    <<<"$(qwen38_mlx_native_registry_identity "$root_dir")"
jq -n --arg source_commit "$(git -C "$root_dir" rev-parse HEAD)" \
    --arg dependency_version "$dependency_version" \
    --arg dependency_source "$dependency_source" \
    --arg dependency_checksum "$dependency_checksum" \
    --arg runner_sha "$(shasum -a 256 "$script_dir/qwen_family_mtp_artifact_matrix.sh" | awk '{print $1}')" \
    --arg contract_sha "$(shasum -a 256 "$script_dir/qwen_family_mtp_artifact_contract.sh" | awk '{print $1}')" \
    --argjson results "$results" '{
      schema:1,verdict:"pass",gate:"qwen-family-native-mtp-compound-matrix",
      source_commit:$source_commit,
      dependency:{name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum},
      evidence:{runner_sha256:$runner_sha,contract_sha256:$contract_sha},
      cells:["qwen35-dense","qwen36-dense","qwen36-moe"],results:$results}
    ' >"$test_dir/matrix.json"
seal_family_fixture "$test_dir"
qwen_family_mtp_validate_matrix_receipt "$test_dir/matrix.json"
qwen_family_mtp_validate_matrix_seal "$test_dir/matrix.json" "$root_dir"

jq '.results[2].proof.shared_embedding_and_exact_head_allocation=false' \
    "$test_dir/matrix.json" >"$test_dir/bad-shared.json"
expect_failure shared-head-substitution qwen_family_mtp_validate_matrix_receipt \
    "$test_dir/bad-shared.json"
jq '.results[0].artifact.sha256=("0"*64)' \
    "$test_dir/matrix.json" >"$test_dir/bad-artifact.json"
expect_failure wrong-artifact qwen_family_mtp_validate_matrix_receipt \
    "$test_dir/bad-artifact.json"

mutation_root="$test_dir/mutations"
mkdir -p "$mutation_root"
log_tamper="$mutation_root/log-tamper"
copy_family_fixture "$test_dir" "$log_tamper"
printf 'tampered\n' >>"$log_tamper/qwen36-moe.log"
expect_failure sealed-log-tamper qwen_family_mtp_validate_matrix_seal \
    "$log_tamper/matrix.json" "$root_dir"

cell_divergence="$mutation_root/cell-divergence"
copy_family_fixture "$test_dir" "$cell_divergence"
jq '.proof.quantized_dispatch_observed=false' \
    "$cell_divergence/qwen35-dense.json" >"$cell_divergence/qwen35-dense.json.tmp"
mv "$cell_divergence/qwen35-dense.json.tmp" "$cell_divergence/qwen35-dense.json"
seal_family_fixture "$cell_divergence"
expect_failure sealed-cell-divergence qwen_family_mtp_validate_matrix_seal \
    "$cell_divergence/matrix.json" "$root_dir"

dependency_tamper="$mutation_root/dependency-tamper"
copy_family_fixture "$test_dir" "$dependency_tamper"
jq '.dependency.checksum=("0"*64)' "$dependency_tamper/matrix.json" \
    >"$dependency_tamper/matrix.json.tmp"
mv "$dependency_tamper/matrix.json.tmp" "$dependency_tamper/matrix.json"
seal_family_fixture "$dependency_tamper"
expect_failure sealed-dependency-tamper qwen_family_mtp_validate_matrix_seal \
    "$dependency_tamper/matrix.json" "$root_dir"

extra_entry="$mutation_root/extra-entry"
copy_family_fixture "$test_dir" "$extra_entry"
printf 'unsealed extra\n' >"$extra_entry/extra.txt"
expect_failure sealed-extra-entry qwen_family_mtp_validate_matrix_seal \
    "$extra_entry/matrix.json" "$root_dir"

runner="$script_dir/qwen_family_mtp_artifact_matrix.sh"
bash -n "$runner"
# shellcheck disable=SC2016
grep -Fq -- '--manifest-path "$root_dir/Cargo.toml"' "$runner" \
    || fail 'family runner does not bind the candidate manifest'
grep -Fq 'block_segmented_native_qwen_family_mtp_artifact_matches_sequential_control' "$runner" \
    || fail 'family runner does not execute the exact native MTP test'
# shellcheck disable=SC2016
grep -Fq 'qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"' "$runner" \
    || fail 'family runner does not reject ambient Cargo configuration'
grep -Fq 'HF2Q_DECODE_MVN=1 HF2Q_DECODE_MV_EXT=0' "$runner" \
    || fail 'family runner does not pin the coherent quantized decode route'

echo 'Qwen family MTP artifact contract: PASS'
