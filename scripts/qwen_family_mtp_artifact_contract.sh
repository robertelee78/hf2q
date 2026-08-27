#!/usr/bin/env bash

# Immutable complete-GGUF cells for the shared Qwen3.5/Qwen3.6 MTP runtime.
# Each file contains the verifier and its one trained MTP block; MTP-only
# overlays and artifacts requantized from another GGUF are intentionally not
# accepted as source-truth cells.

qwen_family_contract_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! declare -F qwen38_validate_evidence_manifest_paths >/dev/null; then
    # shellcheck source=scripts/qwen38_artifact_contract.sh
    source "$qwen_family_contract_dir/qwen38_artifact_contract.sh"
fi

qwen_family_mtp_cells() {
    printf '%s\n' qwen35-dense qwen36-dense qwen36-moe
}

# Tab-separated: cell, repository, revision, relative file, bytes, SHA-256,
# architecture, GGUF file type, target layers, total blocks, variant.
qwen_family_mtp_record() {
    case ${1:-} in
        qwen35-dense)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                qwen35-dense unsloth/Qwen3.5-4B-MTP-GGUF \
                86835bf9949e4d14d6860f7910b1340ad4f271a9 \
                qwen35-dense/Qwen3.5-4B-Q4_K_M.gguf 2834975040 \
                3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630 \
                qwen35 15 32 33 dense
            ;;
        qwen36-dense)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                qwen36-dense froggeric/Qwen3.6-27B-MTP-GGUF \
                704849950ef6410bc877e22a86bb44d2ffc16248 \
                qwen36-dense/Qwen3.6-27B-Q4_K_M-mtp.gguf 16810716064 \
                c0754e3014b4db6668425b33d7b64e92e927210c39b2352eb66083197c2a7722 \
                qwen35 15 64 65 dense
            ;;
        qwen36-moe)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                qwen36-moe unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
                5bc3e238d916f48a861bac2f8a1990a0e9b7e98d \
                qwen36-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf 22663387424 \
                0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b \
                qwen35moe 15 40 41 moe
            ;;
        *)
            echo "Qwen family MTP cell must be qwen35-dense, qwen36-dense, or qwen36-moe" >&2
            return 1
            ;;
    esac
}

qwen_family_mtp_validate_artifact_identity() {
    local cell=$1 actual_bytes=$2 actual_sha=$3
    local _cell _repository _revision _file expected_bytes expected_sha
    local _arch _file_type _target_layers _total_blocks _variant
    IFS=$'\t' read -r _cell _repository _revision _file expected_bytes expected_sha \
        _arch _file_type _target_layers _total_blocks _variant \
        <<<"$(qwen_family_mtp_record "$cell")" || return 1
    [[ "$actual_bytes" == "$expected_bytes" && "$actual_sha" == "$expected_sha" ]] || {
        echo "Qwen family MTP artifact identity mismatch for $cell" >&2
        echo "expected bytes=$expected_bytes sha256=$expected_sha" >&2
        echo "actual   bytes=$actual_bytes sha256=$actual_sha" >&2
        return 1
    }
}

qwen_family_mtp_validate_matrix_receipt() {
    local receipt=$1 cell index=0
    [[ -f "$receipt" && -r "$receipt" && ! -L "$receipt" ]] || return 1
    jq -e '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen-family-native-mtp-compound-matrix"
      and (.source_commit | test("^[0-9a-f]{40}$"))
      and .dependency.name == "mlx-native"
      and (.dependency.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
      and .dependency.source ==
        "registry+https://github.com/rust-lang/crates.io-index"
      and (.dependency.checksum | test("^[0-9a-f]{64}$"))
      and .cells == ["qwen35-dense","qwen36-dense","qwen36-moe"]
      and (.results | map(.cell)) == .cells
      and all(.results[];
        .verdict == "pass"
        and .artifact.file_type == 15
        and .proof.metadata_preflight_before_metal == true
        and .proof.native_storage_without_substitution == true
        and .proof.shared_embedding_and_exact_head_allocation == true
        and .proof.mtp_compound_matches_sequential == true
        and .proof.target_and_mtp_final_bytes_equal == true
        and .proof.quantized_dispatch_observed == true
        and (.log.sha256 | test("^[0-9a-f]{64}$")))
    ' "$receipt" >/dev/null || return 1
    while IFS= read -r cell; do
        local _cell repository revision file bytes sha arch file_type target_layers total_blocks variant
        IFS=$'\t' read -r _cell repository revision file bytes sha arch file_type \
            target_layers total_blocks variant <<<"$(qwen_family_mtp_record "$cell")" || return 1
        jq -e --argjson index "$index" --arg cell "$cell" --arg repository "$repository" \
            --arg revision "$revision" --arg file "$file" --arg sha "$sha" \
            --arg arch "$arch" --arg variant "$variant" --argjson bytes "$bytes" \
            --argjson file_type "$file_type" --argjson target_layers "$target_layers" \
            --argjson total_blocks "$total_blocks" '
          .results[$index]
          | .cell == $cell
          and .repository == $repository and .revision == $revision
          and .artifact.file == $file and .artifact.sha256 == $sha
          and .artifact.bytes == $bytes and .artifact.file_type == $file_type
          and .artifact.architecture == $arch and .artifact.variant == $variant
          and .artifact.target_layers == $target_layers
          and .artifact.total_blocks == $total_blocks
        ' "$receipt" >/dev/null || return 1
        index=$((index + 1))
    done < <(qwen_family_mtp_cells)
}

qwen_family_mtp_validate_matrix_seal() {
    local receipt=$1 source_root=${2:-} receipt_dir evidence result
    local expected_paths actual_paths expected_entries actual_entries
    local cell cell_path log_path expected_cell actual_cell
    local source_identity receipt_identity matrix_sha evidence_sha
    [[ "$(basename "$receipt")" == matrix.json ]] || return 1
    receipt_dir=$(cd "$(dirname "$receipt")" && pwd) || return 1
    evidence="$receipt_dir/evidence.sha256"
    result="$receipt_dir/result.sha256"
    qwen_family_mtp_validate_matrix_receipt "$receipt" || return 1
    if [[ -n "$source_root" ]]; then
        source_identity=$(qwen38_mlx_native_registry_identity "$source_root") \
            || return 1
        receipt_identity=$(jq -er '
          [.dependency.version,.dependency.source,.dependency.checksum] | @tsv
        ' "$receipt") || return 1
        [[ "$receipt_identity" == "$source_identity" ]] || {
            echo "Qwen family MTP receipt dependency identity differs from source" >&2
            return 1
        }
    fi
    [[ -f "$evidence" && ! -L "$evidence" && -f "$result" && ! -L "$result" ]] || return 1
    qwen38_validate_evidence_manifest_paths "$evidence" || return 1
    expected_paths='qwen35-dense.json
qwen35-dense.log
qwen36-dense.json
qwen36-dense.log
qwen36-moe.json
qwen36-moe.log'
    actual_paths=$(awk '{ print substr($0, 67) }' "$evidence" | sort) || return 1
    [[ "$actual_paths" == "$expected_paths" ]] || {
        echo "Qwen family MTP evidence manifest is incomplete or contains extras" >&2
        return 1
    }
    expected_entries='evidence.sha256
matrix.json
qwen35-dense.json
qwen35-dense.log
qwen36-dense.json
qwen36-dense.log
qwen36-moe.json
qwen36-moe.log
result.sha256'
    actual_entries=$(cd "$receipt_dir" \
        && find . -mindepth 1 -maxdepth 1 -print | sed 's#^./##' | sort) \
        || return 1
    [[ "$actual_entries" == "$expected_entries" ]] || {
        echo "Qwen family MTP evidence directory is incomplete or contains extras" >&2
        return 1
    }
    for cell in $(qwen_family_mtp_cells); do
        cell_path="$receipt_dir/$cell.json"
        log_path="$receipt_dir/$cell.log"
        [[ -f "$cell_path" && ! -L "$cell_path" \
            && -f "$log_path" && ! -L "$log_path" ]] || return 1
        expected_cell=$(jq -Sce --arg cell "$cell" \
            '.results[] | select(.cell == $cell)' "$receipt") || return 1
        actual_cell=$(jq -Sce . "$cell_path") || return 1
        [[ "$actual_cell" == "$expected_cell" ]] || return 1
        [[ "$(jq -er '.log.path' "$cell_path")" == "$cell.log" \
            && "$(jq -er '.log.sha256' "$cell_path")" == \
                "$(shasum -a 256 "$log_path" | awk '{print $1}')" ]] || return 1
    done
    [[ "$(awk 'END {print NR}' "$result")" == 2 ]] || return 1
    matrix_sha=$(shasum -a 256 "$receipt" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence" | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$result")" == "$matrix_sha  matrix.json" \
        && "$(sed -n '2p' "$result")" == "$evidence_sha  evidence.sha256" ]] \
        || return 1
    (cd "$receipt_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
        && shasum -a 256 -c result.sha256 >/dev/null) || return 1
    if [[ -n "$source_root" ]]; then
        [[ "$(git -C "$source_root" rev-parse HEAD)" == "$(jq -er .source_commit "$receipt")" ]] \
            || return 1
        jq -e --arg runner "$(shasum -a 256 "$source_root/scripts/qwen_family_mtp_artifact_matrix.sh" | awk '{print $1}')" \
            --arg contract "$(shasum -a 256 "$source_root/scripts/qwen_family_mtp_artifact_contract.sh" | awk '{print $1}')" '
          .evidence.runner_sha256 == $runner
          and .evidence.contract_sha256 == $contract
        ' "$receipt" >/dev/null || return 1
    fi
}
