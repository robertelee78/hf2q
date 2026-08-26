#!/usr/bin/env bash

qwen35_ordinary_contract_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! declare -F qwen38_validate_evidence_manifest_paths >/dev/null; then
    # shellcheck source=scripts/qwen38_artifact_contract.sh
    source "$qwen35_ordinary_contract_dir/qwen38_artifact_contract.sh"
fi

qwen35_validate_ordinary_probe_tsv() {
    local path=$1
    [[ -f "$path" && -r "$path" && ! -L "$path" ]] || return 1
    awk -F '\t' '
      BEGIN {
        expected[1]="schema"; expected[2]="route";
        expected[3]="prefix_tokens"; expected[4]="continuation_tokens";
        expected[5]="prefix_logits_sha256"; expected[6]="prefix_hidden_sha256";
        expected[7]="prefix_cache_sha256"; expected[8]="continuation_logits_sha256";
        expected[9]="continuation_hidden_sha256";
        expected[10]="continuation_cache_sha256";
      }
      NF != 2 || $1 != expected[NR] || seen[$1]++ { invalid=1 }
      NR == 1 && $2 != "1" { invalid=1 }
      NR == 2 && $2 != "ordinary-target-plus-mtp" { invalid=1 }
      NR == 3 && $2 != "33" { invalid=1 }
      NR == 4 && $2 != "3" { invalid=1 }
      NR >= 5 && $2 !~ /^[0-9a-f]{64}$/ { invalid=1 }
      END { exit !(NR == 10 && !invalid) }
    ' "$path"
}

qwen35_validate_ordinary_identity_receipt() {
    local receipt=$1
    [[ -f "$receipt" && -r "$receipt" && ! -L "$receipt" ]] || return 1
    jq -e '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen35-ordinary-main-byte-identity"
      and (.baseline.commit | test("^[0-9a-f]{40}$"))
      and (.candidate.commit | test("^[0-9a-f]{40}$"))
      and .baseline.commit != .candidate.commit
      and all([.baseline.dependency,.candidate.dependency][];
        .name == "mlx-native"
        and (.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
        and .source == "registry+https://github.com/rust-lang/crates.io-index"
        and (.checksum | test("^[0-9a-f]{64}$")))
      and .model.format == "Q4_K_M" and .model.file_type == 15
      and (.model.path | startswith("/"))
      and (.model.sha256 | test("^[0-9a-f]{64}$"))
      and (.model.bytes | numbers) > 0
      and (.model.file_snapshot | length) > 0
      and .route == {speculation:"off",tq_kv:true,mlx_disp_bucket:1,
        mvn:false,mv_ext:true,prefix_tokens:33,continuation_tokens:3}
      and .proof == {same_probe_on_both_commits:true,
        ordinary_target_and_mtp_direct:true,exact_probe_receipts_equal:true,
        exact_f32_logits_equal:true,exact_target_hidden_equal:true,
        exact_physical_target_and_mtp_cache_bytes_equal:true}
      and (.identity | keys | sort) ==
        (["continuation_cache_sha256","continuation_hidden_sha256",
          "continuation_logits_sha256","prefix_cache_sha256",
          "prefix_hidden_sha256","prefix_logits_sha256"] | sort)
      and all(.identity[]; test("^[0-9a-f]{64}$"))
      and all([.evidence.runner_sha256,.evidence.contract_sha256,
        .evidence.probe_module_sha256,.evidence.probe_patch_sha256][];
        test("^[0-9a-f]{64}$"))
    ' "$receipt" >/dev/null
}

qwen35_validate_ordinary_identity_seal() {
    local receipt=$1 baseline_root=${2:-} candidate_root=${3:-}
    local receipt_dir evidence result expected_paths actual_paths
    local expected_entries actual_entries receipt_sha evidence_sha
    local baseline_tsv candidate_tsv source_identity receipt_identity
    [[ "$(basename "$receipt")" == receipt.json ]] || return 1
    receipt_dir=$(cd "$(dirname "$receipt")" && pwd) || return 1
    evidence="$receipt_dir/evidence.sha256"
    result="$receipt_dir/result.sha256"
    [[ -f "$evidence" && ! -L "$evidence" && -f "$result" && ! -L "$result" ]] \
        || return 1
    qwen35_validate_ordinary_identity_receipt "$receipt" || return 1
    qwen38_validate_evidence_manifest_paths "$evidence" || return 1
    expected_paths='baseline.log
baseline.probe.tsv
candidate.log
candidate.probe.tsv'
    actual_paths=$(awk '{ print substr($0, 67) }' "$evidence" | sort) || return 1
    [[ "$actual_paths" == "$expected_paths" ]] || return 1
    expected_entries='baseline.log
baseline.probe.tsv
candidate.log
candidate.probe.tsv
evidence.sha256
receipt.json
result.sha256'
    actual_entries=$(cd "$receipt_dir" \
        && find . -mindepth 1 -maxdepth 1 -print | sed 's#^./##' | sort) \
        || return 1
    [[ "$actual_entries" == "$expected_entries" ]] || return 1
    baseline_tsv="$receipt_dir/baseline.probe.tsv"
    candidate_tsv="$receipt_dir/candidate.probe.tsv"
    qwen35_validate_ordinary_probe_tsv "$baseline_tsv" || return 1
    qwen35_validate_ordinary_probe_tsv "$candidate_tsv" || return 1
    cmp -s "$baseline_tsv" "$candidate_tsv" || return 1
    while IFS=$'\t' read -r key value; do
        case "$key" in
            *_sha256)
                [[ "$(jq -er --arg key "$key" '.identity[$key]' "$receipt")" == "$value" ]] \
                    || return 1
                ;;
        esac
    done <"$candidate_tsv"
    [[ "$(awk 'END { print NR }' "$result")" == 2 ]] || return 1
    receipt_sha=$(shasum -a 256 "$receipt" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence" | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$result")" == "$receipt_sha  receipt.json" \
        && "$(sed -n '2p' "$result")" == "$evidence_sha  evidence.sha256" ]] \
        || return 1
    (cd "$receipt_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
        && shasum -a 256 -c result.sha256 >/dev/null) || return 1
    if [[ -n "$baseline_root" || -n "$candidate_root" ]]; then
        [[ -n "$baseline_root" && -n "$candidate_root" ]] || return 1
        for side in baseline candidate; do
            if [[ "$side" == baseline ]]; then
                source_root=$baseline_root
            else
                source_root=$candidate_root
            fi
            [[ "$(git -C "$source_root" rev-parse HEAD)" == \
                "$(jq -er ".$side.commit" "$receipt")" ]] || return 1
            source_identity=$(qwen38_mlx_native_registry_identity "$source_root") || return 1
            receipt_identity=$(jq -er --arg side "$side" \
                '.[$side].dependency | [.version,.source,.checksum] | @tsv' \
                "$receipt") || return 1
            [[ "$source_identity" == "$receipt_identity" ]] || return 1
        done
        jq -e \
            --arg runner "$(shasum -a 256 "$candidate_root/scripts/qwen35_ordinary_byte_identity.sh" | awk '{print $1}')" \
            --arg contract "$(shasum -a 256 "$candidate_root/scripts/qwen35_ordinary_identity_contract.sh" | awk '{print $1}')" \
            --arg module "$(shasum -a 256 "$candidate_root/scripts/probes/qwen35_ordinary_byte_identity_probe.rs" | awk '{print $1}')" \
            --arg patch "$(shasum -a 256 "$candidate_root/scripts/probes/qwen35_ordinary_byte_identity_module.patch" | awk '{print $1}')" '
          .evidence.runner_sha256 == $runner
          and .evidence.contract_sha256 == $contract
          and .evidence.probe_module_sha256 == $module
          and .evidence.probe_patch_sha256 == $patch
        ' "$receipt" >/dev/null || return 1
    fi
}
