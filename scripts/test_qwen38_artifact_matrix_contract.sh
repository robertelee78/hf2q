#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"

qwen_module="$script_dir/../src/inference/models/qwen35/mod.rs"
qwen_model="$script_dir/../src/inference/models/qwen35/model.rs"
qwen_engine="$script_dir/../src/serve/api/engine_qwen35.rs"
qwen_launcher="$script_dir/serve_qwen38_opencode.sh"

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

seal_four_fixture() {
    local root=$1 path
    : >"$root/evidence.sha256.tmp"
    while IFS= read -r path; do
        printf '%s  %s\n' \
            "$(shasum -a 256 "$root/$path" | awk '{print $1}')" "$path" \
            >>"$root/evidence.sha256.tmp"
    done < <(find "$root" -maxdepth 1 -type f \
        ! -name matrix.json ! -name evidence.sha256 \
        ! -name evidence.sha256.tmp ! -name result.sha256 \
        -exec basename {} \; | sort)
    mv "$root/evidence.sha256.tmp" "$root/evidence.sha256"
    printf '%s  matrix.json\n%s  evidence.sha256\n' \
        "$(shasum -a 256 "$root/matrix.json" | awk '{print $1}')" \
        "$(shasum -a 256 "$root/evidence.sha256" | awk '{print $1}')" \
        >"$root/result.sha256"
}

seal_physical_fixture() {
    local root=$1 path
    : >"$root/evidence.sha256.tmp"
    while IFS= read -r path; do
        case "$path" in
            matrix.json|evidence.sha256|evidence.sha256.tmp|result.sha256)
                continue
                ;;
        esac
        printf '%s  %s\n' \
            "$(shasum -a 256 "$root/$path" | awk '{print $1}')" "$path" \
            >>"$root/evidence.sha256.tmp"
    done < <(cd "$root" && find . -type f -print | sed 's#^./##' | sort)
    mv "$root/evidence.sha256.tmp" "$root/evidence.sha256"
    printf '%s  matrix.json\n%s  evidence.sha256\n' \
        "$(shasum -a 256 "$root/matrix.json" | awk '{print $1}')" \
        "$(shasum -a 256 "$root/evidence.sha256" | awk '{print $1}')" \
        >"$root/result.sha256"
}

grep -Fq 'mlx_native::ggml_routing_policy_from_environment()' "$qwen_module" \
    || fail "Qwen artifacts do not resolve the shared native routing policy"
if grep -Fq 'apply_qwen38_routing_defaults' "$qwen_module"; then
    fail "Qwen3.8 still rewrites the shared coherent routing defaults"
fi
grep -Fq 'preflight_qwen35_gguf_with_routing(gguf, &cfg, routing)' "$qwen_model" \
    || fail "Qwen preflight and runtime do not share one frozen routing policy"
if grep -Fq 'std::env::set_var' "$qwen_engine"; then
    fail "Qwen engine mutates process environment during model load"
fi
if grep -Eq '^export HF2Q_DECODE_(MVN|MV_EXT)=' "$qwen_launcher"; then
    fail "Qwen3.8 launcher synthesizes process-global model routing"
fi
set +e
inexact_launcher_output=$(
    QWEN38_SPECULATION=auto HF2Q_DECODE_MV_EXT=1 \
        bash "$qwen_launcher" 2>&1
)
inexact_launcher_status=$?
set -e
[[ "$inexact_launcher_status" == 3 ]] \
    || fail "Qwen3.8 launcher did not reject inexact mv_ext speculation"
[[ "$inexact_launcher_output" == *"not exact across scalar and width-four target routes"* ]] \
    || fail "Qwen3.8 launcher did not explain the inexact mv_ext rejection"

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-matrix-contract.XXXXXX")
cleanup() {
    case "$test_dir" in
        "${TMPDIR:-/tmp}"/hf2q-qwen38-matrix-contract.*)
            rm -rf -- "$test_dir"
            ;;
        *)
            echo "refusing unsafe test cleanup: $test_dir" >&2
            ;;
    esac
}
trap cleanup EXIT

dependency_root="$test_dir/dependency-valid"
test_cargo_home="$test_dir/cargo-home"
mkdir -p "$dependency_root" "$test_cargo_home"
cat >"$dependency_root/Cargo.toml" <<'EOF'
[dependencies]
mlx-native = "=9.8.7"
EOF
cat >"$dependency_root/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "mlx-native"
version = "9.8.7"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
EOF
dependency_identity=$(qwen38_mlx_native_registry_identity "$dependency_root")
[[ "$dependency_identity" == \
  "9.8.7"$'\t'"registry+https://github.com/rust-lang/crates.io-index"$'\t'\
"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ]] \
  || fail "valid registry dependency identity was not preserved"
qwen38_reject_cargo_configuration "$dependency_root" "$test_cargo_home"
(
    export RUSTC_WRAPPER=sccache
    expect_failure cargo-rustc-wrapper qwen38_reject_cargo_configuration \
        "$dependency_root" "$test_cargo_home"
)
(
    export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS=-Ctarget-cpu=native
    expect_failure cargo-target-rustflags qwen38_reject_cargo_configuration \
        "$dependency_root" "$test_cargo_home"
)
for override in \
    CARGO_TARGET_DIR=/var/tmp/foreign-target \
    CARGO_BUILD_TARGET=aarch64-apple-darwin \
    CARGO_PROFILE_RELEASE_LTO=thin \
    RUSTUP_TOOLCHAIN=nightly \
    RUSTC_BOOTSTRAP=1; do
    name=${override%%=*}
    value=${override#*=}
    (
        export "$name=$value"
        expect_failure "cargo-environment-$name" qwen38_reject_cargo_configuration \
            "$dependency_root" "$test_cargo_home"
    )
done

dependency_comment_only_config="$test_dir/dependency-comment-only-config"
cp -R "$dependency_root" "$dependency_comment_only_config"
mkdir -p "$dependency_comment_only_config/.cargo"
printf '# no active Cargo overrides\n\n  # comments are inert\n' \
    >"$dependency_comment_only_config/.cargo/config.toml"
qwen38_reject_cargo_configuration \
    "$dependency_comment_only_config" "$test_cargo_home"

dependency_wrong_source="$test_dir/dependency-wrong-source"
cp -R "$dependency_root" "$dependency_wrong_source"
sed 's#registry+https://github.com/rust-lang/crates.io-index#path+file:///tmp/mlx-native#' \
    "$dependency_wrong_source/Cargo.lock" \
    >"$dependency_wrong_source/Cargo.lock.tmp"
mv "$dependency_wrong_source/Cargo.lock.tmp" "$dependency_wrong_source/Cargo.lock"
expect_failure dependency-path-source qwen38_mlx_native_registry_identity \
    "$dependency_wrong_source"

dependency_missing_checksum="$test_dir/dependency-missing-checksum"
cp -R "$dependency_root" "$dependency_missing_checksum"
sed '/^checksum = /d' "$dependency_missing_checksum/Cargo.lock" \
    >"$dependency_missing_checksum/Cargo.lock.tmp"
mv "$dependency_missing_checksum/Cargo.lock.tmp" \
    "$dependency_missing_checksum/Cargo.lock"
expect_failure dependency-missing-checksum qwen38_mlx_native_registry_identity \
    "$dependency_missing_checksum"

dependency_version_mismatch="$test_dir/dependency-version-mismatch"
cp -R "$dependency_root" "$dependency_version_mismatch"
sed 's/version = "9.8.7"/version = "9.8.6"/' \
    "$dependency_version_mismatch/Cargo.lock" \
    >"$dependency_version_mismatch/Cargo.lock.tmp"
mv "$dependency_version_mismatch/Cargo.lock.tmp" \
    "$dependency_version_mismatch/Cargo.lock"
expect_failure dependency-version-mismatch qwen38_mlx_native_registry_identity \
    "$dependency_version_mismatch"

dependency_nonexact_manifest="$test_dir/dependency-nonexact-manifest"
cp -R "$dependency_root" "$dependency_nonexact_manifest"
sed 's/"=9.8.7"/"9.8.7"/' "$dependency_nonexact_manifest/Cargo.toml" \
    >"$dependency_nonexact_manifest/Cargo.toml.tmp"
mv "$dependency_nonexact_manifest/Cargo.toml.tmp" \
    "$dependency_nonexact_manifest/Cargo.toml"
expect_failure dependency-nonexact-manifest qwen38_mlx_native_registry_identity \
    "$dependency_nonexact_manifest"

dependency_local_config="$test_dir/dependency-local-config"
cp -R "$dependency_root" "$dependency_local_config"
mkdir -p "$dependency_local_config/.cargo"
printf '[patch.crates-io]\nmlx-native = { path = "/tmp/mlx-native" }\n' \
    >"$dependency_local_config/.cargo/config.toml"
expect_failure dependency-local-config qwen38_reject_cargo_configuration \
    "$dependency_local_config" "$test_cargo_home"

dependency_ancestor="$test_dir/dependency-ancestor"
mkdir -p "$dependency_ancestor/.cargo" "$dependency_ancestor/nested"
printf '[patch.crates-io]\nmlx-native = { path = "/tmp/mlx-native" }\n' \
    >"$dependency_ancestor/.cargo/config.toml"
expect_failure dependency-ancestor-config qwen38_reject_cargo_configuration \
    "$dependency_ancestor/nested" "$test_cargo_home"

cargo_home_with_config="$test_dir/cargo-home-with-config"
mkdir -p "$cargo_home_with_config"
printf '[source.crates-io]\nreplace-with = "local"\n' \
    >"$cargo_home_with_config/config.toml"
expect_failure dependency-cargo-home-config qwen38_reject_cargo_configuration \
    "$dependency_root" "$cargo_home_with_config"

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

snapshot_fixture="$test_dir/artifact-snapshot.gguf"
printf '%s' 'immutable artifact' >"$snapshot_fixture"
snapshot_identity=$(qwen38_artifact_snapshot "$snapshot_fixture")
qwen38_validate_artifact_snapshot_unchanged \
    "$snapshot_fixture" "$snapshot_identity"
printf '%s' ' mutation' >>"$snapshot_fixture"
expect_failure artifact-snapshot-mutation \
    qwen38_validate_artifact_snapshot_unchanged \
    "$snapshot_fixture" "$snapshot_identity"

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
grep -Fq -- '--locked --bin hf2q "$trajectory_test"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not run the trajectory test target"
# shellcheck disable=SC2016
grep -Fq -- '--locked --bin hf2q "$segmented_test"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not run the segmented test target"
grep -Fq "grep -Ec 'test result: ok\\. 1 passed; 0 failed; 0 ignored;'" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not require all three full-model tests to pass"
for route_mapping in \
    'expected_mvn_qtype=Q4_K' \
    'expected_mvn_qtype=Q5_K' \
    'expected_mvn_qtype=Q6_K'; do
    grep -Fq "$route_mapping" \
        "$script_dir/qwen38_four_position_artifact_matrix.sh" \
        || fail "four-position matrix omits exact route mapping: $route_mapping"
done
grep -Fq 'expected_mvn_kernel=kernel_mul_mv_ext_q5_K_f32_r1_4' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "Q5_K matrix cell does not require the canonical q4x4 route"
# shellcheck disable=SC2016
grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not bind the Q5_K route per cell"
# shellcheck disable=SC2016
grep -Fq '[QWEN38_DENSE_ROUTE] qtype=$expected_mvn_qtype kernel=$expected_mvn_kernel' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not require the production dispatch canary"

expect_failure unknown-format qwen38_artifact_record Q3_K_M
expect_failure case-folded-format qwen38_artifact_record q5_k_m

pin=$(qwen38_pinned_peer_commit)
[[ "$pin" =~ ^[0-9a-f]{40}$ ]] || fail "pinned peer commit is not exact: $pin"
[[ "$pin" == "bf942164697d2d62c2237a17b677dc2c017ea8e7" ]] \
    || fail "pinned peer commit drifted from regenerated fixture authority: $pin"
qwen38_validate_pinned_peer_commit "$pin"
expect_failure wrong-peer-commit qwen38_validate_pinned_peer_commit \
    0000000000000000000000000000000000000000

four_cells="$test_dir/four-cells.jsonl"
physical_cells="$test_dir/physical-cells.jsonl"
matched_cells="$test_dir/matched-cells.jsonl"
four_seal="$test_dir/four-seal"
mkdir -p "$four_seal"
: >"$four_cells"
: >"$physical_cells"
: >"$matched_cells"
for format in "${expected_formats[@]}"; do
    IFS=$'\t' read -r _format file bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    printf 'four-position proof for %s\n' "$format" \
        >"$four_seal/$format_slug.log"
    if [[ "$format" == Q5_K_M ]]; then
        printf '%s\n' \
          '[QWEN38_DENSE_ROUTE] qtype=Q5_K kernel=kernel_mul_mv_ext_q5_K_f32_r1_4' \
          >>"$four_seal/$format_slug.log"
    fi
    log_sha256=$(shasum -a 256 "$four_seal/$format_slug.log" \
        | awk '{print $1}')
    case "$format" in
        Q4_K_M) exact_mvn_qtype=Q4_K ;;
        Q5_K_M) exact_mvn_qtype=Q5_K ;;
        Q6_K) exact_mvn_qtype=Q6_K ;;
        *) exact_mvn_qtype= ;;
    esac
    q5k_canonical_q4x4=true
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" \
        --arg log_path "$format_slug.log" --arg log_sha256 "$log_sha256" \
        --arg exact_mvn_qtype "$exact_mvn_qtype" \
        --argjson q5k_canonical_q4x4 "$q5k_canonical_q4x4" '{
          schema:2,verdict:"pass",format:$format,
          artifact:{path:("/models/" + $file),sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          proof:{native_storage_without_substitution:true,
            four_position_matches_scalar:true,
            eight_token_post_batch_handoff_matches_scalar:true,
            stable_boundary_compound_matches_split:true,
            rectangular_state_and_continuation_match_scalar:true,
            exact_mvn_width_four_qtype:(if $exact_mvn_qtype == "" then null else $exact_mvn_qtype end),
            decode_route:{mlx_disp_bucket:1,mvn:true,mv_ext:false,
              q5k_canonical_q4x4:$q5k_canonical_q4x4}},
          log:{path:$log_path,sha256:$log_sha256}
        }' >>"$four_cells"
    width_results=$(jq -nc '[1,2,4,8,16] | map(. as $width | {
      schema:2,verdict:"pass",width:$width,
      runtime:{routing:{q5k_canonical_q4x4:true},
        frozen_qwen_policy_log_count:1},
      request:{exact_scalar_replay_per_lane:true},
      metrics:{scheduler_max_width:$width,target_body_max_width:$width,
        target_head_max_width:$width,target_forwards_delta:1,
        target_body_rows_delta:$width,target_head_rows_delta:$width,
        command_buffers_created_delta:1,command_buffer_submissions_delta:1},
      clients:[range(1;$width+1) | {lane:.,scalar_parity:true}]
    })')
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" --argjson results "$width_results" '{
          schema:2,verdict:"pass",
          binary:{sha256:("2" * 64)},
          model:{format:$format,file:$file,
            repository:"jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            revision:"0a72776892f98db49381fdf69f4b9982222ec9dc",
            sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          workload:{widths:[1,2,4,8,16],max_tokens:64,
            exact_scalar_replay_per_lane:true,server_restart_per_width:true,
            kv_cache_budget_bytes:51539607552,
            routing:{decode_mvn:1,decode_mv_ext:0,
              q5k_canonical_q4x4:true}},
          results:$results
        }' >>"$physical_cells"
    jq -nc --arg format "$format" --arg file "$file" \
        --arg sha256 "$sha256" --argjson bytes "$bytes" \
        --argjson file_type "$file_type" --arg peer_commit "$pin" '{
          schema:5,verdict:"pass",
          launch_settings:{
            schema:2,
            hf2q:{dense_decode_mvn:1,dense_decode_mv_ext:0,
              dense_q5k_canonical_q4x4:1,
              speculation:"adaptive-history-then-mtp-cost-gated",
              kv_cache:"tq-kv",kv_cache_budget_bytes:51539607552,
              context_tokens_per_slot:262144},
            reference:{speculation:"fixed-k3-mtp",kv_cache_k:"q8_0",
              kv_cache_v:"q8_0",context_tokens_total:262144}},
          hf2q:{commit:("3" * 40),binary_sha256:("4" * 64),
            effective_routing_policy:{dense_decode_mvn:1,
              dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
            speculation:"adaptive-history-then-mtp-cost-gated"},
          reference:{commit:$peer_commit,speculation:"fixed-k3-mtp"},
          model:{format:$format,file:$file,
            repository:"jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            revision:"0a72776892f98db49381fdf69f4b9982222ec9dc",
            sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          quality:{code:{evaluator_tests_passed:true},
            repeat:{exact_expected_content:true}},
          calibration:{trial_logs:24,
            host_contention:{policy:"process-group-cpu-v2",
              maximum_foreign_cpu_percent:100,
              owner_scope:"release-gate-process-group",owner_pgid:100,
              continuous:true}},
          stability:{stable:true},acceptance:{minimum_hf2q_ratio:1},
          code:{hf2q_over_reference:1.01},repeat:{hf2q_over_reference:1.01}
        }' >>"$matched_cells"
done

matrix_runner_sha=$(shasum -a 256 "$script_dir/qwen38_physical_multislot_matrix.sh" \
    | awk '{print $1}')
four_runner_sha=$(shasum -a 256 \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" | awk '{print $1}')
gate_runner_sha=$(shasum -a 256 "$script_dir/qwen38_physical_multislot_gate.sh" \
    | awk '{print $1}')
physical_contract_sha=$(shasum -a 256 \
    "$script_dir/qwen38_physical_multislot_contract.sh" | awk '{print $1}')
artifact_contract_sha=$(shasum -a 256 "$script_dir/qwen38_artifact_contract.sh" \
    | awk '{print $1}')
jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg runner_sha "$four_runner_sha" \
    --arg contract_sha "$artifact_contract_sha" '{
      schema:1,verdict:"pass",gate:"qwen38-four-position-artifact-matrix",
      source_commit:("5" * 40),repository:$repository,revision:$revision,
      dependency:{name:"mlx-native",version:"9.8.7",
        source:"registry+https://github.com/rust-lang/crates.io-index",
        checksum:("a" * 64)},
      evidence:{runner_sha256:$runner_sha,
        artifact_contract_sha256:$contract_sha},
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:.
    }' "$four_cells" >"$test_dir/four.json"
cp "$test_dir/four.json" "$four_seal/matrix.json"
for index in "${!expected_formats[@]}"; do
    slug=$(printf '%s' "${expected_formats[$index]}" \
        | tr '[:upper:]' '[:lower:]')
    jq -c --argjson index "$index" '.results[$index]' \
        "$test_dir/four.json" >"$four_seal/$slug.json"
done
seal_four_fixture "$four_seal"
four_matrix_sha=$(shasum -a 256 "$four_seal/matrix.json" | awk '{print $1}')
qwen38_copy_four_position_matrix_seal \
    "$four_seal/matrix.json" "$test_dir/four-position" "$dependency_root"
jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg four_matrix_sha "$four_matrix_sha" '{
      schema:2,verdict:"pass",gate:"qwen38-artifact-physical-width-matrix",
      repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
      widths:[1,2,4,8,16],
      workload:{max_tokens:64,kv_cache_budget_bytes:51539607552,
        routing:{decode_mvn:1,decode_mv_ext:0,q5k_canonical_q4x4:true}},
      route_proof:{four_position_matrix_sha256:$four_matrix_sha,
        q5k:{qtype:"Q5_K",width:4,
          kernel:"kernel_mul_mv_ext_q5_K_f32_r1_4",actual_dispatch:true}},
      results:.
    }' "$physical_cells" >"$test_dir/physical.json"
jq --arg matrix "$matrix_runner_sha" --arg gate "$gate_runner_sha" \
    --arg contract "$physical_contract_sha" \
    --arg artifact "$artifact_contract_sha" \
    '.evidence={matrix_runner_sha256:$matrix,gate_runner_sha256:$gate,
      physical_contract_sha256:$contract,artifact_contract_sha256:$artifact}' \
    "$test_dir/physical.json" >"$test_dir/physical-with-evidence.json"
mkdir -p "$test_dir/physical-seal/raw"
cp "$test_dir/physical-with-evidence.json" "$test_dir/physical-seal/matrix.json"
qwen38_copy_four_position_matrix_seal \
    "$four_seal/matrix.json" "$test_dir/physical-seal/four-position" \
    "$dependency_root"
for format in "${expected_formats[@]}"; do
    slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    for width in 1 2 4 8 16; do
        mkdir -p "$test_dir/physical-seal/$slug/width-$width"
        printf '%s\n' \
          'INFO frozen Qwen GGML routing policy dense_decode_mvn=true dense_decode_mv_ext=false dense_q5k_canonical_q4x4=true' \
          >"$test_dir/physical-seal/$slug/width-$width/server.log"
    done
done
printf '%s\n' physical-proof >"$test_dir/physical-seal/raw/proof.txt"
seal_physical_fixture "$test_dir/physical-seal"
cp "$test_dir/physical-with-evidence.json" "$test_dir/physical.json"
jq -s --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg peer_commit "$pin" '{
      schema:2,verdict:"pass",gate:"qwen38-matched-peer-artifact-matrix",
      repository:$repository,revision:$revision,pinned_peer_commit:$peer_commit,
      hf2q_effective_routing_policy:{dense_decode_mvn:1,
        dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:.
    }' "$matched_cells" >"$test_dir/matched.json"

qwen38_validate_four_position_matrix_receipt "$test_dir/four.json"
qwen38_validate_four_position_matrix_seal \
    "$four_seal/matrix.json" "$dependency_root"
four_log_tamper="$test_dir/four-log-tamper"
cp -R "$four_seal" "$four_log_tamper"
printf 'tampered\n' >>"$four_log_tamper/q6_k.log"
expect_failure four-seal-log-tamper qwen38_validate_four_position_matrix_seal \
    "$four_log_tamper/matrix.json" "$dependency_root"

four_missing_q5_canary="$test_dir/four-missing-q5-canary"
cp -R "$four_seal" "$four_missing_q5_canary"
sed '/kernel_mul_mv_ext_q5_K_f32_r1_4/d' \
    "$four_missing_q5_canary/q5_k_m.log" \
    >"$four_missing_q5_canary/q5_k_m.log.tmp"
mv "$four_missing_q5_canary/q5_k_m.log.tmp" \
    "$four_missing_q5_canary/q5_k_m.log"
jq --arg sha "$(shasum -a 256 "$four_missing_q5_canary/q5_k_m.log" \
    | awk '{print $1}')" '.log.sha256 = $sha' \
    "$four_missing_q5_canary/q5_k_m.json" \
    >"$four_missing_q5_canary/q5_k_m.json.tmp"
mv "$four_missing_q5_canary/q5_k_m.json.tmp" \
    "$four_missing_q5_canary/q5_k_m.json"
jq --slurpfile cell "$four_missing_q5_canary/q5_k_m.json" \
    '(.results[] | select(.format == "Q5_K_M")) = $cell[0]' \
    "$four_missing_q5_canary/matrix.json" \
    >"$four_missing_q5_canary/matrix.json.tmp"
mv "$four_missing_q5_canary/matrix.json.tmp" \
    "$four_missing_q5_canary/matrix.json"
seal_four_fixture "$four_missing_q5_canary"
expect_failure four-seal-missing-q5-canary \
    qwen38_validate_four_position_matrix_seal \
    "$four_missing_q5_canary/matrix.json" "$dependency_root"

four_cell_divergence="$test_dir/four-cell-divergence"
cp -R "$four_seal" "$four_cell_divergence"
jq '.proof.four_position_matches_scalar = false' \
    "$four_cell_divergence/q5_k_m.json" \
    >"$four_cell_divergence/q5_k_m.json.tmp"
mv "$four_cell_divergence/q5_k_m.json.tmp" \
    "$four_cell_divergence/q5_k_m.json"
seal_four_fixture "$four_cell_divergence"
expect_failure four-seal-cell-divergence \
    qwen38_validate_four_position_matrix_seal \
    "$four_cell_divergence/matrix.json" "$dependency_root"

four_segmented_divergence="$test_dir/four-segmented-divergence"
cp -R "$four_seal" "$four_segmented_divergence"
jq '.proof.stable_boundary_compound_matches_split = false' \
    "$four_segmented_divergence/q8_0.json" \
    >"$four_segmented_divergence/q8_0.json.tmp"
mv "$four_segmented_divergence/q8_0.json.tmp" \
    "$four_segmented_divergence/q8_0.json"
seal_four_fixture "$four_segmented_divergence"
expect_failure four-seal-segmented-divergence \
    qwen38_validate_four_position_matrix_seal \
    "$four_segmented_divergence/matrix.json" "$dependency_root"

four_rectangular_divergence="$test_dir/four-rectangular-divergence"
cp -R "$four_seal" "$four_rectangular_divergence"
jq '.proof.rectangular_state_and_continuation_match_scalar = false' \
    "$four_rectangular_divergence/q6_k.json" \
    >"$four_rectangular_divergence/q6_k.json.tmp"
mv "$four_rectangular_divergence/q6_k.json.tmp" \
    "$four_rectangular_divergence/q6_k.json"
seal_four_fixture "$four_rectangular_divergence"
expect_failure four-seal-rectangular-divergence \
    qwen38_validate_four_position_matrix_seal \
    "$four_rectangular_divergence/matrix.json" "$dependency_root"

four_missing_evidence="$test_dir/four-missing-evidence"
cp -R "$four_seal" "$four_missing_evidence"
rm "$four_missing_evidence/q4_k_m.json"
seal_four_fixture "$four_missing_evidence"
expect_failure four-seal-missing-evidence \
    qwen38_validate_four_position_matrix_seal \
    "$four_missing_evidence/matrix.json" "$dependency_root"

four_symlink_evidence="$test_dir/four-symlink-evidence"
cp -R "$four_seal" "$four_symlink_evidence"
rm "$four_symlink_evidence/q8_0.log"
ln -s "$four_seal/q8_0.log" "$four_symlink_evidence/q8_0.log"
expect_failure four-seal-symlink-evidence \
    qwen38_validate_four_position_matrix_seal \
    "$four_symlink_evidence/matrix.json" "$dependency_root"

four_extra_entry="$test_dir/four-extra-entry"
cp -R "$four_seal" "$four_extra_entry"
ln -s "$four_seal/q8_0.log" "$four_extra_entry/unsealed-extra"
expect_failure four-seal-extra-entry \
    qwen38_validate_four_position_matrix_seal \
    "$four_extra_entry/matrix.json" "$dependency_root"

four_path_dependency="$test_dir/four-path-dependency"
cp -R "$four_seal" "$four_path_dependency"
jq '.dependency.source = "path+file:///tmp/mlx-native"' \
    "$four_path_dependency/matrix.json" \
    >"$four_path_dependency/matrix.json.tmp"
mv "$four_path_dependency/matrix.json.tmp" \
    "$four_path_dependency/matrix.json"
seal_four_fixture "$four_path_dependency"
expect_failure four-seal-path-dependency \
    qwen38_validate_four_position_matrix_seal \
    "$four_path_dependency/matrix.json" "$dependency_root"

four_wrong_dependency_checksum="$test_dir/four-wrong-dependency-checksum"
cp -R "$four_seal" "$four_wrong_dependency_checksum"
jq '.dependency.checksum = ("b" * 64)' \
    "$four_wrong_dependency_checksum/matrix.json" \
    >"$four_wrong_dependency_checksum/matrix.json.tmp"
mv "$four_wrong_dependency_checksum/matrix.json.tmp" \
    "$four_wrong_dependency_checksum/matrix.json"
seal_four_fixture "$four_wrong_dependency_checksum"
expect_failure four-seal-wrong-dependency-checksum \
    qwen38_validate_four_position_matrix_seal \
    "$four_wrong_dependency_checksum/matrix.json" "$dependency_root"
qwen38_validate_physical_matrix_receipt "$test_dir/physical.json"
qwen38_validate_physical_matrix_seal "$test_dir/physical-seal/matrix.json"
qwen38_copy_physical_matrix_seal "$test_dir/physical-seal/matrix.json" \
    "$test_dir/physical-copy"
qwen38_validate_physical_matrix_seal "$test_dir/physical-copy/matrix.json"
mv "$test_dir/physical-seal" "$test_dir/physical-seal-away"
qwen38_validate_physical_matrix_seal "$test_dir/physical-copy/matrix.json"
mv "$test_dir/physical-seal-away" "$test_dir/physical-seal"
printf '%s\n' tampered >>"$test_dir/physical-copy/raw/proof.txt"
expect_failure physical-seal-evidence-tamper qwen38_validate_physical_matrix_seal \
    "$test_dir/physical-copy/matrix.json"
mkdir -p "$test_dir/physical-symlink"
ln -s "$test_dir/physical-seal/matrix.json" \
    "$test_dir/physical-symlink/matrix.json"
cp "$test_dir/physical-seal/evidence.sha256" \
    "$test_dir/physical-symlink/evidence.sha256"
cp "$test_dir/physical-seal/result.sha256" \
    "$test_dir/physical-symlink/result.sha256"
expect_failure physical-symlink-receipt qwen38_validate_physical_matrix_seal \
    "$test_dir/physical-symlink/matrix.json"
physical_false_frozen="$test_dir/physical-false-frozen"
cp -R "$test_dir/physical-seal" "$physical_false_frozen"
sed 's/dense_q5k_canonical_q4x4=true/dense_q5k_canonical_q4x4=false/' \
    "$physical_false_frozen/q5_k_m/width-4/server.log" \
    >"$physical_false_frozen/q5_k_m/width-4/server.log.tmp"
mv "$physical_false_frozen/q5_k_m/width-4/server.log.tmp" \
    "$physical_false_frozen/q5_k_m/width-4/server.log"
seal_physical_fixture "$physical_false_frozen"
expect_failure physical-false-frozen-policy \
    qwen38_validate_physical_matrix_seal \
    "$physical_false_frozen/matrix.json"
qwen38_validate_matched_peer_matrix_receipt "$test_dir/matched.json"

jq 'del(.results[4])' "$test_dir/four.json" >"$test_dir/four-missing-format.json"
expect_failure four-missing-format qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-missing-format.json"
jq '.results[1].proof.exact_mvn_width_four_qtype = null' \
    "$test_dir/four.json" >"$test_dir/four-missing-route-proof.json"
expect_failure four-missing-route-proof qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-missing-route-proof.json"
jq '.results[2].proof.exact_mvn_width_four_qtype = "Q4_K"' \
    "$test_dir/four.json" >"$test_dir/four-wrong-q5-route-proof.json"
expect_failure four-wrong-q5-route-proof qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-wrong-q5-route-proof.json"
jq '.results[2].proof.decode_route.q5k_canonical_q4x4 = false' \
    "$test_dir/four.json" >"$test_dir/four-q5-policy-disabled.json"
expect_failure four-q5-policy-disabled qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-q5-policy-disabled.json"
jq '.results[4].proof.decode_route.mv_ext = true' \
    "$test_dir/four.json" >"$test_dir/four-ambient-route.json"
expect_failure four-ambient-route qwen38_validate_four_position_matrix_receipt \
    "$test_dir/four-ambient-route.json"
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
jq '.results[].workload.max_tokens = 63' \
    "$test_dir/physical.json" >"$test_dir/physical-noncanonical-max-tokens.json"
expect_failure physical-noncanonical-max-tokens \
    qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-noncanonical-max-tokens.json"
jq '.workload.max_tokens = 63' \
    "$test_dir/physical.json" >"$test_dir/physical-outer-max-tokens.json"
expect_failure physical-outer-max-tokens qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-outer-max-tokens.json"
jq '.results[2].workload.kv_cache_budget_bytes += 1' \
    "$test_dir/physical.json" >"$test_dir/physical-kv-budget-drift.json"
expect_failure physical-kv-budget-drift qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-kv-budget-drift.json"
jq '.results[3].workload.routing.decode_mvn = 0' \
    "$test_dir/physical.json" >"$test_dir/physical-mvn-drift.json"
expect_failure physical-mvn-drift qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-mvn-drift.json"
jq '.results[4].workload.routing.decode_mv_ext = 1' \
    "$test_dir/physical.json" >"$test_dir/physical-mv-ext-drift.json"
expect_failure physical-mv-ext-drift qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-mv-ext-drift.json"
jq '.workload.routing.q5k_canonical_q4x4 = false' \
    "$test_dir/physical.json" >"$test_dir/physical-q5-policy-disabled.json"
expect_failure physical-q5-policy-disabled \
    qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-q5-policy-disabled.json"
jq '.results[2].workload.routing.q5k_canonical_q4x4 = false' \
    "$test_dir/physical.json" >"$test_dir/physical-q5-child-drift.json"
expect_failure physical-q5-child-drift qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-q5-child-drift.json"
jq '.route_proof.q5k.kernel = "kernel_mul_mv_q5_K_f32_mN_r1_4"' \
    "$test_dir/physical.json" >"$test_dir/physical-q5-route-lie.json"
expect_failure physical-q5-route-lie qwen38_validate_physical_matrix_receipt \
    "$test_dir/physical-q5-route-lie.json"
jq '.results[4].verdict = "skip"' \
    "$test_dir/matched.json" >"$test_dir/matched-skipped-cell.json"
expect_failure matched-skipped-cell qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-skipped-cell.json"
jq '.schema = 1' "$test_dir/matched.json" \
    >"$test_dir/matched-old-matrix-schema.json"
expect_failure matched-old-matrix-schema \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-old-matrix-schema.json"
jq '.results[2].schema = 4' "$test_dir/matched.json" \
    >"$test_dir/matched-old-child-schema.json"
expect_failure matched-old-child-schema \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-old-child-schema.json"
jq '.results[2].calibration.trial_logs = 16' "$test_dir/matched.json" \
    >"$test_dir/matched-stale-contention-log-count.json"
expect_failure matched-stale-contention-log-count \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-stale-contention-log-count.json"
jq '.results[2].calibration.host_contention.policy = "process-group-v1"' \
    "$test_dir/matched.json" >"$test_dir/matched-stale-contention-policy.json"
expect_failure matched-stale-contention-policy \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-stale-contention-policy.json"
jq '.results[2].calibration.host_contention.maximum_foreign_cpu_percent = 101' \
    "$test_dir/matched.json" >"$test_dir/matched-weak-contention-threshold.json"
expect_failure matched-weak-contention-threshold \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-weak-contention-threshold.json"
jq '.results[2].calibration.host_contention.owner_pgid = 0' \
    "$test_dir/matched.json" >"$test_dir/matched-invalid-contention-owner.json"
expect_failure matched-invalid-contention-owner \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-invalid-contention-owner.json"
jq '.results[2].calibration.host_contention.continuous = false' \
    "$test_dir/matched.json" >"$test_dir/matched-noncontinuous-contention.json"
expect_failure matched-noncontinuous-contention \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-noncontinuous-contention.json"
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
jq '.results[2].launch_settings.hf2q.dense_decode_mvn = 0' \
    "$test_dir/matched.json" >"$test_dir/matched-route-lie.json"
expect_failure matched-route-lie qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-route-lie.json"
jq '.results[2].launch_settings.hf2q.dense_q5k_canonical_q4x4 = 0' \
    "$test_dir/matched.json" >"$test_dir/matched-q5-launch-lie.json"
expect_failure matched-q5-launch-lie \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-q5-launch-lie.json"
jq '.results[2].hf2q.effective_routing_policy
      .dense_q5k_canonical_q4x4 = 0' \
    "$test_dir/matched.json" >"$test_dir/matched-q5-effective-lie.json"
expect_failure matched-q5-effective-lie \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-q5-effective-lie.json"
jq '.hf2q_effective_routing_policy.dense_q5k_canonical_q4x4 = 0' \
    "$test_dir/matched.json" >"$test_dir/matched-q5-outer-lie.json"
expect_failure matched-q5-outer-lie \
    qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-q5-outer-lie.json"
jq '.results[2].launch_settings.reference.speculation = "shipping-auto"' \
    "$test_dir/matched.json" >"$test_dir/matched-policy-lie.json"
expect_failure matched-policy-lie qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-policy-lie.json"
jq '.results[2].launch_settings.reference.kv_cache_k = "tq-kv"' \
    "$test_dir/matched.json" >"$test_dir/matched-cache-lie.json"
expect_failure matched-cache-lie qwen38_validate_matched_peer_matrix_receipt \
    "$test_dir/matched-cache-lie.json"

# Build a complete self-contained matched-physical matrix whose copied
# physical evidence and five child seals are all reopened by the validator.
matched_root="$test_dir/matched-physical"
mkdir -p "$matched_root/artifacts"
qwen38_copy_physical_matrix_seal "$test_dir/physical-seal/matrix.json" \
    "$matched_root/physical-proof"
physical_sha=$(shasum -a 256 "$matched_root/physical-proof/matrix.json" \
    | awk '{print $1}')
physical_binary_sha=$(jq -er '.results[0].binary.sha256' \
    "$matched_root/physical-proof/matrix.json")

speculation=$(jq -nc '[
  {schema:1,engine:"hf2q",trial:1,group:"code",
    policy:"adaptive-history-then-mtp-cost-gated",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:1,equivalent_ordinary_seconds:0.5},
  {schema:1,engine:"hf2q",trial:1,group:"repeat",
    policy:"adaptive-history-then-mtp-cost-gated",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:1,equivalent_ordinary_seconds:0.5},
  {schema:1,engine:"reference",trial:2,group:"code",policy:"fixed-k3-mtp",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:0,equivalent_ordinary_seconds:0},
  {schema:1,engine:"reference",trial:2,group:"repeat",policy:"fixed-k3-mtp",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:0,equivalent_ordinary_seconds:0},
  {schema:1,engine:"reference",trial:3,group:"code",policy:"fixed-k3-mtp",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:0,equivalent_ordinary_seconds:0},
  {schema:1,engine:"reference",trial:3,group:"repeat",policy:"fixed-k3-mtp",proof_pass:true,
    proof_mode:"accepted-proposals",disable_reason:null,proposals:2,
    drafted_tokens:4,accepted_tokens:3,cost_disabled_generations:0,
    measured_round_seconds:0,equivalent_ordinary_seconds:0},
  {schema:1,engine:"hf2q",trial:4,group:"code",
    policy:"adaptive-history-then-mtp-cost-gated",proof_pass:true,
    proof_mode:"measured-cost-disabled",disable_reason:"measured_cost_unprofitable",
    proposals:2,drafted_tokens:4,accepted_tokens:0,cost_disabled_generations:1,
    measured_round_seconds:1,equivalent_ordinary_seconds:0.5},
  {schema:1,engine:"hf2q",trial:4,group:"repeat",
    policy:"adaptive-history-then-mtp-cost-gated",proof_pass:true,
    proof_mode:"measured-cost-disabled",disable_reason:"measured_cost_unprofitable",
    proposals:2,drafted_tokens:4,accepted_tokens:0,cost_disabled_generations:1,
    measured_round_seconds:1,equivalent_ordinary_seconds:0.5}
]')
outer_results='[]'
children='[]'
for index in "${!expected_formats[@]}"; do
    format=${expected_formats[$index]}
    slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    IFS=$'\t' read -r _format file bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    cells='[]'
    for width in 1 2 4 8 16; do
        clients=$(jq -nc --argjson width "$width" \
            '[range(1;$width+1) | {lane:.,scalar_parity:true}]')
        cell=$(jq -nc --argjson width "$width" --argjson clients "$clients" \
            --argjson speculation "$speculation" '{
              schema:2,verdict:"pass",width:$width,
              hf2q_effective_routing_policy:{dense_decode_mvn:1,
                dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
              samples:{hf2q:2,reference:2},
              scalar_replay:{hf2q:true,reference:true},
              physical_proof:{width:$width,mode:"ordinary-target-speculation-off",
                seal_validated:true,scheduler_max_width:$width,
                target_body_max_width:$width,target_head_max_width:$width,
                command_buffers_created_delta:1,
                command_buffer_submissions_delta:1,clients:$clients},
              speculation:{hf2q_policy:"adaptive-history-then-mtp-cost-gated",
                reference_policy:"fixed-k3-mtp",waves:$speculation},
              acceptance:{minimum_hf2q_ratio:1},
              code:{width:$width,group:"code",quality_pass:true,
                measurement_consistent:true,api_concurrency_pass:true,
                token_accounting:{pass:true,
                  cross_engine_prompt_equality_required:true,
                  raw_api_completion_equality_required_within_engine:true,
                  cross_engine_completion_equality_required:false,
                  cross_engine_semantic_completion_equality_required:false},
                stability:{stable:true,
                  observed_band_dominance:true},
                hf2q_over_reference_comparison_rate:1.01},
              repeat:{width:$width,group:"repeat",quality_pass:true,
                measurement_consistent:true,api_concurrency_pass:true,
                token_accounting:{pass:true,
                  cross_engine_prompt_equality_required:true,
                  raw_api_completion_equality_required_within_engine:true,
                  cross_engine_completion_equality_required:false,
                  cross_engine_semantic_completion_equality_required:true,
                  semantic_tokenization_sha256:("d"*64)},
                stability:{stable:true,
                  observed_band_dominance:true},
                hf2q_over_reference_comparison_rate:1.01,
                reference_over_hf2q_p95_wall:1.01,
                semantic_ttft:{required:true,stable:true,
                  observed_band_dominance:true,reference_over_hf2q_p95:1.01}}}')
        cells=$(jq -nc --argjson prior "$cells" --argjson cell "$cell" \
            '$prior + [$cell]')
    done
    child_dir="$matched_root/artifacts/$slug"
    mkdir -p "$child_dir"
    jq -n --arg format "$format" --arg file "$file" --arg sha "$sha256" \
        --argjson bytes "$bytes" --argjson file_type "$file_type" \
        --arg pin "$pin" --arg physical_sha "$physical_sha" \
        --argjson cells "$cells" '{
          schema:2,verdict:"pass",gate:"qwen38-matched-physical-abba",
          hf2q:{commit:("3"*40),binary_sha256:("2"*64),
            effective_routing_policy:{dense_decode_mvn:1,
              dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1}},
          reference:{commit:$pin,binary_sha256:("6"*64)},
          physical_matrix_sha256:$physical_sha,
          model:{format:$format,file:$file,
            repository:"jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            revision:"0a72776892f98db49381fdf69f4b9982222ec9dc",
            sha256:$sha,bytes:$bytes,gguf_file_type:$file_type},
          workload:{widths:[1,2,4,8,16],
            trial_order:["hf2q","reference","reference","hf2q"],
            speculation:{hf2q:"adaptive-history-then-mtp-cost-gated",
              reference:"fixed-k3-mtp"},
            cache_settings:{
              hf2q:{format:"tq-kv",budget_bytes:51539607552,
                context_tokens_per_slot:262144},
              reference:{k_format:"q8_0",v_format:"q8_0",
                context_tokens_total:262144}},
            scalar_replay_per_lane:true,
            repeat_semantic_tokenization:{receipt_sha256:("d"*64),
              completion_tokens:100,unit:"canonical-semantic-output-token"},
            reference_parallelism_matches_width:true},
          acceptance:{minimum_hf2q_ratio:1,
            maximum_launch_skew_seconds:0.1},
          host_contention:{policy:"process-group-cpu-v2",
            maximum_foreign_cpu_percent:100,
            owner_scope:"release-gate-process-group",owner_pgid:100,
            continuous:true},results:$cells}' \
        >"$child_dir/summary.json"
    printf '%s\n' child-proof >"$child_dir/payload.txt"
    printf '%s  payload.txt\n' \
        "$(shasum -a 256 "$child_dir/payload.txt" | awk '{print $1}')" \
        >"$child_dir/evidence.sha256"
    printf '%s  summary.json\n%s  evidence.sha256\n' \
        "$(shasum -a 256 "$child_dir/summary.json" | awk '{print $1}')" \
        "$(shasum -a 256 "$child_dir/evidence.sha256" | awk '{print $1}')" \
        >"$child_dir/result.sha256"
    outer_results=$(jq -nc --argjson prior "$outer_results" \
        --slurpfile child "$child_dir/summary.json" '$prior + [$child[0]]')
    child_seal=$(jq -nc --arg format "$format" --arg path "artifacts/$slug" \
        --arg summary "$(shasum -a 256 "$child_dir/summary.json" | awk '{print $1}')" \
        --arg evidence "$(shasum -a 256 "$child_dir/evidence.sha256" | awk '{print $1}')" \
        --arg result "$(shasum -a 256 "$child_dir/result.sha256" | awk '{print $1}')" '{
          format:$format,path:$path,summary_sha256:$summary,
          evidence_manifest_sha256:$evidence,result_seal_sha256:$result}')
    children=$(jq -nc --argjson prior "$children" --argjson child "$child_seal" \
        '$prior + [$child]')
done
jq -n --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" --arg pin "$pin" \
    --arg physical_sha "$physical_sha" --arg binary "$physical_binary_sha" \
    --argjson results "$outer_results" --argjson children "$children" '{
      schema:2,verdict:"pass",gate:"qwen38-matched-physical-artifact-matrix",
      repository:$repository,revision:$revision,pinned_reference_commit:$pin,
      hf2q_effective_routing_policy:{dense_decode_mvn:1,
        dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
      widths:[1,2,4,8,16],
      workload:{
        speculation:{hf2q:"adaptive-history-then-mtp-cost-gated",
          reference:"fixed-k3-mtp"},
        cache_settings:{
          hf2q:{format:"tq-kv",budget_bytes:51539607552,
            context_tokens_per_slot:262144},
          reference:{k_format:"q8_0",v_format:"q8_0",
            context_tokens_total:262144}}},
      acceptance:{maximum_launch_skew_seconds:0.1},
      physical_matrix:{sha256:$physical_sha,seal_validated:true,
        self_contained_path:"physical-proof/matrix.json",
        gate:"qwen38-artifact-physical-width-matrix",binary_sha256:$binary},
      evidence:{script_sha256:("7"*64),contract_sha256:("8"*64),
        artifact_contract_sha256:("9"*64),child_results_sealed:true,
        children:$children},results:$results}' >"$matched_root/summary.json.tmp"
qwen38_validate_matched_physical_matrix_receipt "$matched_root/summary.json.tmp"

# Re-seal coherent child mutations so rejection proves the outer validator
# reads the v2 contention contract rather than merely noticing a stale hash.
contention_child="$matched_root/artifacts/q5_k_m"
cp "$contention_child/summary.json" "$test_dir/q5-child-summary.good"
cp "$contention_child/result.sha256" "$test_dir/q5-child-result.good"
for mutation in stale-policy weak-threshold invalid-owner noncontinuous; do
    case "$mutation" in
      stale-policy) filter='.host_contention.policy="process-group-v1"' ;;
      weak-threshold) filter='.host_contention.maximum_foreign_cpu_percent=101' ;;
      invalid-owner) filter='.host_contention.owner_pgid=0' ;;
      noncontinuous) filter='.host_contention.continuous=false' ;;
    esac
    jq "$filter" "$test_dir/q5-child-summary.good" \
        >"$contention_child/summary.json"
    printf '%s  summary.json\n%s  evidence.sha256\n' \
        "$(shasum -a 256 "$contention_child/summary.json" | awk '{print $1}')" \
        "$(shasum -a 256 "$contention_child/evidence.sha256" | awk '{print $1}')" \
        >"$contention_child/result.sha256"
    mutated_summary_sha=$(shasum -a 256 "$contention_child/summary.json" \
        | awk '{print $1}')
    mutated_result_sha=$(shasum -a 256 "$contention_child/result.sha256" \
        | awk '{print $1}')
    jq --slurpfile child "$contention_child/summary.json" \
        --arg summary "$mutated_summary_sha" --arg result "$mutated_result_sha" '
      .results[2]=$child[0]
      | .evidence.children[2].summary_sha256=$summary
      | .evidence.children[2].result_seal_sha256=$result
    ' "$matched_root/summary.json.tmp" \
        >"$matched_root/resealed-contention-$mutation.json"
    expect_failure "matched-physical-resealed-contention-$mutation" \
        qwen38_validate_matched_physical_matrix_receipt \
        "$matched_root/resealed-contention-$mutation.json"
done
cp "$test_dir/q5-child-summary.good" "$contention_child/summary.json"
cp "$test_dir/q5-child-result.good" "$contention_child/result.sha256"

for mutation in old-matrix-schema old-child-schema old-cell-schema \
    unsealed-physical wrong-physical-path lane-zero fake-cost-disable \
    physical-join child-digest child-embedding weak-threshold raw-api-scope \
    cross-engine-raw-scope semantic-scope semantic-tokenization outer-kv-budget \
    outer-reference-policy outer-reference-cache outer-launch-skew \
    child-kv-budget child-reference-policy child-launch-skew \
    outer-q5-policy child-q5-policy cell-q5-policy; do
    case "$mutation" in
      old-matrix-schema) filter='.schema=1' ;;
      old-child-schema) filter='.results[2].schema=1' ;;
      old-cell-schema) filter='.results[2].results[3].schema=1' ;;
      unsealed-physical) filter='.physical_matrix.seal_validated=false' ;;
      wrong-physical-path) filter='.physical_matrix.self_contained_path="matrix.json"' ;;
      lane-zero) filter='.results[2].results[3].physical_proof.clients[0].lane=0' ;;
      fake-cost-disable) filter='.results[2].results[3].speculation.waves[7]
        .measured_round_seconds=0' ;;
      physical-join) filter='.results[2].results[3]
        .physical_proof.command_buffer_submissions_delta=2' ;;
      child-digest) filter='.evidence.children[1].summary_sha256=("0"*64)' ;;
      child-embedding) filter='.results[1].unsealed_fabrication=true' ;;
      weak-threshold) filter='.results[2].results[3]
        .acceptance.minimum_hf2q_ratio=0.5' ;;
      raw-api-scope) filter='.results[2].results[3].code.token_accounting
        .raw_api_completion_equality_required_within_engine=false' ;;
      cross-engine-raw-scope) filter='.results[2].results[3].repeat
        .token_accounting.cross_engine_completion_equality_required=true' ;;
      semantic-scope) filter='.results[2].results[3].repeat.token_accounting
        .cross_engine_semantic_completion_equality_required=false' ;;
      semantic-tokenization) filter='.results[2].results[3].repeat
        .token_accounting.semantic_tokenization_sha256=("e"*64)' ;;
      outer-kv-budget) filter='.workload.cache_settings.hf2q.budget_bytes += 1' ;;
      outer-reference-policy) filter='.workload.speculation.reference="shipping-auto"' ;;
      outer-reference-cache) filter='.workload.cache_settings.reference.k_format="tq-kv"' ;;
      outer-launch-skew) filter='.acceptance.maximum_launch_skew_seconds=0.2' ;;
      child-kv-budget) filter='.results[2].workload.cache_settings.hf2q.budget_bytes += 1' ;;
      child-reference-policy) filter='.results[2].workload.speculation.reference="shipping-auto"' ;;
      child-launch-skew) filter='.results[2].acceptance.maximum_launch_skew_seconds=0.2' ;;
      outer-q5-policy) filter='.hf2q_effective_routing_policy
        .dense_q5k_canonical_q4x4=0' ;;
      child-q5-policy) filter='.results[2].hf2q.effective_routing_policy
        .dense_q5k_canonical_q4x4=0' ;;
      cell-q5-policy) filter='.results[2].results[3]
        .hf2q_effective_routing_policy.dense_q5k_canonical_q4x4=0' ;;
    esac
    jq "$filter" "$matched_root/summary.json.tmp" \
        >"$matched_root/$mutation.json"
    expect_failure "matched-physical-$mutation" \
        qwen38_validate_matched_physical_matrix_receipt \
        "$matched_root/$mutation.json"
done

cp "$matched_root/artifacts/q4_k_m/result.sha256" \
    "$matched_root/artifacts/q4_k_m/result.sha256.good"
payload_sha=$(shasum -a 256 "$matched_root/artifacts/q4_k_m/payload.txt" \
    | awk '{print $1}')
printf '%s  payload.txt\n%s  payload.txt\n' "$payload_sha" "$payload_sha" \
    >"$matched_root/artifacts/q4_k_m/result.sha256"
malformed_result_sha=$(shasum -a 256 \
    "$matched_root/artifacts/q4_k_m/result.sha256" | awk '{print $1}')
jq --arg sha "$malformed_result_sha" \
    '.evidence.children[1].result_seal_sha256=$sha' \
    "$matched_root/summary.json.tmp" >"$matched_root/malformed-child-result.json"
expect_failure malformed-child-result \
    qwen38_validate_matched_physical_matrix_receipt \
    "$matched_root/malformed-child-result.json"
mv "$matched_root/artifacts/q4_k_m/result.sha256.good" \
    "$matched_root/artifacts/q4_k_m/result.sha256"

# Manifest validation is independent of shasum's path handling.
printf '%s  ../outside\n' "$(printf outside | shasum -a 256 | awk '{print $1}')" \
    >"$test_dir/path-traversal.sha256"
expect_failure manifest-path-traversal qwen38_validate_evidence_manifest_paths \
    "$test_dir/path-traversal.sha256"
printf '%s  result.sha256\n' \
    "$(printf reserved | shasum -a 256 | awk '{print $1}')" \
    >"$test_dir/reserved-manifest.sha256"
expect_failure manifest-reserved-control qwen38_validate_evidence_manifest_paths \
    "$test_dir/reserved-manifest.sha256"
printf '%s\n' tampered >>"$matched_root/physical-proof/raw/proof.txt"
expect_failure copied-physical-tamper \
    qwen38_validate_matched_physical_matrix_receipt \
    "$matched_root/summary.json.tmp"

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
bash -n "$script_dir/qwen38_matched_physical_matrix.sh"
grep -Fq 'qwen38_artifact_formats' \
    "$script_dir/qwen38_matched_physical_matrix.sh" \
    || fail 'matched physical matrix does not enumerate the artifact catalog'
grep -Fq 'qwen38_copy_physical_matrix_seal' \
    "$script_dir/qwen38_matched_physical_matrix.sh" \
    || fail 'matched physical matrix does not copy the sealed physical proof'
grep -Fq "qwen38_validate_four_position_matrix_receipt \"\$OUT_DIR/matrix.json.tmp\"" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail "four-position matrix does not execute its fail-closed receipt validator"
grep -Fq 'Reopen every qualified artifact after all five cells' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not perform a final all-artifact reopen'
grep -Fq 'qwen38_validate_artifact_snapshot_unchanged' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not revalidate final artifact snapshots'
grep -Fq "qwen38_reject_cargo_configuration \"\$root_dir\" \"\$GATE_CARGO_HOME\"" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not reject ignored Cargo configuration'
# shellcheck disable=SC2016
grep -Fq 'CARGO_HOME="$GATE_CARGO_HOME"' \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not build inside the validated Cargo home'
grep -Fq "qwen38_mlx_native_registry_identity \"\$root_dir\"" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not bind the mlx-native registry identity'
grep -Fq "GIT_COMMIT_SHA=\"\$source_commit\"" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not bind test builds to the source commit'
# shellcheck disable=SC2016
[[ "$(grep -Fc -- '--manifest-path "$root_dir/Cargo.toml"' \
  "$script_dir/qwen38_four_position_artifact_matrix.sh")" == 5 ]] \
    || fail 'every four-position cargo process must bind the candidate manifest'

fake_cargo_bin="$test_dir/fake-cargo-bin"
foreign_cwd="$test_dir/foreign-cwd"
fake_cargo_log="$test_dir/fake-cargo.args"
mkdir -p "$fake_cargo_bin" "$foreign_cwd"
# shellcheck disable=SC2016
printf '%s\n' '#!/bin/sh' 'printf "%s\\n" "$@" >"$QWEN38_FAKE_CARGO_LOG"' \
    >"$fake_cargo_bin/cargo"
chmod +x "$fake_cargo_bin/cargo"
(
    export PATH="$fake_cargo_bin:$PATH"
    export QWEN38_FAKE_CARGO_LOG="$fake_cargo_log"
    cd "$foreign_cwd"
    qwen38_cargo_test_from_source "$root_dir" --locked --bin hf2q source-binding-canary
)
[[ "$(sed -n '1p' "$fake_cargo_log")" == test \
    && "$(sed -n '2p' "$fake_cargo_log")" == --manifest-path \
    && "$(sed -n '3p' "$fake_cargo_log")" == "$root_dir/Cargo.toml" ]] \
    || fail 'foreign-CWD Cargo canary did not bind the candidate manifest'
grep -Fq "qwen38_validate_four_position_matrix_seal \"\$OUT_DIR/matrix.json\"" \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" \
    || fail 'four-position matrix does not reopen its complete final seal'
grep -Fq "qwen38_validate_physical_matrix_receipt \"\$OUT_DIR/matrix.json.tmp\"" \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    || fail "physical matrix does not execute its fail-closed receipt validator"
# shellcheck disable=SC2016
for binding in \
    'MAX_TOKENS="$QWEN38_PHYSICAL_MAX_TOKENS"' \
    'KV_CACHE_BUDGET_BYTES="$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES"' \
    'HF2Q_DECODE_MVN="$QWEN38_PHYSICAL_DECODE_MVN"' \
    'HF2Q_DECODE_MV_EXT="$QWEN38_PHYSICAL_DECODE_MV_EXT"' \
    'HF2Q_Q5K_CANONICAL_Q4X4="$QWEN38_PHYSICAL_Q5K_CANONICAL_Q4X4"'; do
    grep -Fq "$binding" "$script_dir/qwen38_physical_multislot_matrix.sh" \
        || fail "physical matrix does not bind canonical child policy: $binding"
done
# shellcheck disable=SC2016
grep -Fq 'FOUR_POSITION_MATRIX_RECEIPT=${FOUR_POSITION_MATRIX_RECEIPT:?' \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    || fail 'physical matrix does not require the sealed four-position route proof'
grep -Fq 'qwen38_copy_four_position_matrix_seal' \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    || fail 'physical matrix does not join the sealed four-position route proof'
grep -Fq "qwen38_validate_physical_matrix_seal \"\$OUT_DIR/matrix.json\"" \
    "$script_dir/qwen38_physical_multislot_matrix.sh" \
    || fail 'physical matrix does not reopen its final seal'
grep -Fq "qwen38_validate_matched_physical_matrix_receipt \"\$OUT_DIR/summary.json.tmp\"" \
    "$script_dir/qwen38_matched_physical_matrix.sh" \
    || fail 'matched physical matrix does not execute its receipt validator'
grep -Fq "matched_physical_validate_reopened_matrix \"\$OUT_DIR\"" \
    "$script_dir/qwen38_matched_physical_matrix.sh" \
    || fail 'matched physical matrix does not semantically reopen its final seal'
# shellcheck disable=SC2016
grep -Fq 'MAX_LAUNCH_SKEW_SECONDS="$MAX_LAUNCH_SKEW_SECONDS"' \
    "$script_dir/qwen38_matched_physical_matrix.sh" \
    || fail 'matched physical matrix does not bind child launch skew'
grep -Fq "qwen38_validate_matched_peer_matrix_receipt \"\$OUT_DIR/matrix.json.tmp\"" \
    "$script_dir/qwen38_matched_peer_matrix.sh" \
    || fail "matched matrix does not execute its fail-closed receipt validator"

bash -n "$script_dir/qwen38_artifact_contract.sh"
bash -n "$script_dir/qwen38_matched_reference_abba.sh"
bash -n "$script_dir/qwen38_physical_multislot_gate.sh"

echo "Qwen3.8 artifact matrix contract: PASS"
