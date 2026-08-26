#!/usr/bin/env bash
set -euo pipefail

# Full-model native-storage and four-position trajectory proof for every
# qualified Qwen3.8 artifact. Each cargo test process unloads its model before
# the next format starts; every K-quant cell also proves its selected native
# width-four projection route.

MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
GATE_CARGO_HOME=${GATE_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-qwen-exact-cargo-home}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"

for command in awk cargo find git grep jq mv sed shasum sort stat tr; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$MODEL_ROOT" == /* && "$OUT_DIR" == /* && "$GATE_CARGO_HOME" == /* ]] || {
    echo "MODEL_ROOT, OUT_DIR, and GATE_CARGO_HOME must be absolute paths" >&2
    exit 2
}
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
case "$GATE_CARGO_HOME/" in
    "$root_dir"/*|"$OUT_DIR"/*)
        echo "GATE_CARGO_HOME must be outside the source and evidence trees" >&2
        exit 2
        ;;
esac
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "four-position artifact-matrix output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
source_commit=$(git -C "$root_dir" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ \
    && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] || {
    echo "four-position artifact matrix requires a clean exact source tree" >&2
    exit 2
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
    <<<"$(qwen38_mlx_native_registry_identity "$root_dir")"

# Refuse a partial or mismatched catalog before loading the first cell.
artifact_sha256s=()
artifact_snapshots=()
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file bytes _expected_sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    [[ -f "$model_path" && -r "$model_path" ]] || {
        echo "qualified $format artifact is missing: $model_path" >&2
        exit 2
    }
    actual_bytes=$(stat -f '%z' "$model_path" 2>/dev/null \
        || stat -c '%s' "$model_path")
    actual_sha256=$(shasum -a 256 "$model_path" | awk '{print $1}')
    qwen38_validate_artifact_identity \
        "$format" "$actual_sha256" "$actual_bytes" "$file_type"
    artifact_sha256s+=("$actual_sha256")
    artifact_snapshots+=("$(qwen38_artifact_snapshot "$model_path")")
done

mkdir -p "$OUT_DIR"
cell_paths=()
trajectory_test='inference::models::qwen35::spec_decode::tests::qwen38_real_four_position_normal_forward_parity'
segmented_test='inference::models::qwen35::forward_gpu::tests::block_segmented_native_qwen38_artifact_matches_sequential_control'
rectangular_test='inference::models::qwen35::forward_gpu::tests::rectangular_target_native_qwen38_format_matches_scalar_state_and_continuation'
artifact_index=0
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file bytes expected_sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    actual_sha256=${artifact_sha256s[$artifact_index]}
    current_snapshot=$(qwen38_artifact_snapshot "$model_path")
    [[ "$current_snapshot" == "${artifact_snapshots[$artifact_index]}" ]] || {
        echo "$format artifact changed after matrix preflight" >&2
        exit 1
    }
    qwen38_validate_artifact_identity \
        "$format" "$actual_sha256" "$bytes" "$file_type"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    log_path="$OUT_DIR/$format_slug.log"
    q5k_canonical_q4x4=1
    case "$format" in
        Q4_K_M)
            expected_mvn_qtype=Q4_K
            expected_mvn_kernel=kernel_mul_mv_q4_K_f32_mN_r1_4
            ;;
        Q5_K_M)
            expected_mvn_qtype=Q5_K
            expected_mvn_kernel=kernel_mul_mv_ext_q5_K_f32_r1_4
            ;;
        Q6_K)
            expected_mvn_qtype=Q6_K
            expected_mvn_kernel=kernel_mul_mv_q6_K_f32_mN_r1_4
            ;;
        *)
            expected_mvn_qtype=
            expected_mvn_kernel=
            ;;
    esac
    if [[ -n "$expected_mvn_qtype" ]]; then
        HF2Q_TEST_QWEN38_FORMAT="$format" \
        HF2Q_TEST_QWEN38_GGUF="$model_path" \
        HF2Q_TEST_QWEN38_EXPECT_MVN_QTYPE="$expected_mvn_qtype" \
        MLX_DISP_BUCKET=1 \
        HF2Q_DECODE_MVN=1 \
        HF2Q_DECODE_MV_EXT=0 \
        HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4" \
        CARGO_HOME="$GATE_CARGO_HOME" \
        GIT_COMMIT_SHA="$source_commit" \
            cargo test --manifest-path "$root_dir/Cargo.toml" \
                --locked --bin hf2q "$trajectory_test" \
                -- --ignored --exact --nocapture >"$log_path" 2>&1
        HF2Q_TEST_QWEN38_FORMAT="$format" \
        HF2Q_TEST_QWEN38_GGUF="$model_path" \
        HF2Q_TEST_QWEN38_EXPECT_MVN_QTYPE="$expected_mvn_qtype" \
        MLX_DISP_BUCKET=1 \
        HF2Q_DECODE_MVN=1 \
        HF2Q_DECODE_MV_EXT=0 \
        HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4" \
        CARGO_HOME="$GATE_CARGO_HOME" \
        GIT_COMMIT_SHA="$source_commit" \
            cargo test --manifest-path "$root_dir/Cargo.toml" \
                --locked --bin hf2q "$segmented_test" \
                -- --ignored --exact --nocapture >>"$log_path" 2>&1
    else
        env -u HF2Q_TEST_QWEN38_EXPECT_MVN_QTYPE \
            HF2Q_TEST_QWEN38_FORMAT="$format" \
            HF2Q_TEST_QWEN38_GGUF="$model_path" \
            MLX_DISP_BUCKET=1 \
            HF2Q_DECODE_MVN=1 \
            HF2Q_DECODE_MV_EXT=0 \
            HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4" \
            CARGO_HOME="$GATE_CARGO_HOME" \
            GIT_COMMIT_SHA="$source_commit" \
            cargo test --manifest-path "$root_dir/Cargo.toml" \
                --locked --bin hf2q "$trajectory_test" \
                -- --ignored --exact --nocapture >"$log_path" 2>&1
        env -u HF2Q_TEST_QWEN38_EXPECT_MVN_QTYPE \
            HF2Q_TEST_QWEN38_FORMAT="$format" \
            HF2Q_TEST_QWEN38_GGUF="$model_path" \
            MLX_DISP_BUCKET=1 \
            HF2Q_DECODE_MVN=1 \
            HF2Q_DECODE_MV_EXT=0 \
            HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4" \
            CARGO_HOME="$GATE_CARGO_HOME" \
            GIT_COMMIT_SHA="$source_commit" \
            cargo test --manifest-path "$root_dir/Cargo.toml" \
                --locked --bin hf2q "$segmented_test" \
                -- --ignored --exact --nocapture >>"$log_path" 2>&1
    fi
    env -u HF2Q_TEST_QWEN38_EXPECT_MVN_QTYPE \
        HF2Q_TEST_QWEN38_FORMAT="$format" \
        HF2Q_TEST_QWEN38_GGUF="$model_path" \
        MLX_DISP_BUCKET=1 \
        HF2Q_DECODE_MVN=1 \
        HF2Q_DECODE_MV_EXT=0 \
        HF2Q_Q5K_CANONICAL_Q4X4="$q5k_canonical_q4x4" \
        CARGO_HOME="$GATE_CARGO_HOME" \
        GIT_COMMIT_SHA="$source_commit" \
        cargo test --manifest-path "$root_dir/Cargo.toml" \
            --locked --bin hf2q "$rectangular_test" \
            -- --ignored --exact --nocapture >>"$log_path" 2>&1
    [[ "$(grep -Ec 'test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log_path")" == 3 ]] || {
        echo "$format did not execute exactly three passing full-model gates" >&2
        sed -n '1,260p' "$log_path" >&2
        exit 1
    }
    if [[ -n "$expected_mvn_qtype" ]]; then
        grep -Fq \
            "[QWEN38_DENSE_ROUTE] qtype=$expected_mvn_qtype kernel=$expected_mvn_kernel" \
            "$log_path" || {
            echo "$format did not publish its selected dense dispatch canary" >&2
            exit 1
        }
    fi
    current_snapshot=$(qwen38_artifact_snapshot "$model_path")
    [[ "$current_snapshot" == "${artifact_snapshots[$artifact_index]}" ]] || {
        echo "$format artifact changed during its full-model proof" >&2
        exit 1
    }
    cell_path="$OUT_DIR/$format_slug.json"
    jq -n \
        --arg format "$format" \
        --arg model_path "$model_path" \
        --arg sha256 "$actual_sha256" \
        --argjson bytes "$bytes" \
        --argjson file_type "$file_type" \
        --arg log_path "$(basename "$log_path")" \
        --arg log_sha256 "$(shasum -a 256 "$log_path" | awk '{print $1}')" \
        --arg exact_mvn_qtype "$expected_mvn_qtype" \
        --argjson q5k_canonical_q4x4 true '{
          schema:2,verdict:"pass",format:$format,
          artifact:{path:$model_path,sha256:$sha256,bytes:$bytes,
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
        }' >"$cell_path.tmp"
    mv "$cell_path.tmp" "$cell_path"
    cell_paths+=("$cell_path")
    artifact_index=$((artifact_index + 1))
done

# Reopen every qualified artifact after all five cells. The initial full hash
# is bound to an inode/size/time snapshot, so this catches catalog mutation
# after an earlier format completed without rereading 132 GiB of weights.
artifact_index=0
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes _sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    qwen38_validate_artifact_snapshot_unchanged \
        "$model_path" "${artifact_snapshots[$artifact_index]}" || {
        echo "$format artifact changed before final matrix sealing" >&2
        exit 1
    }
    current_bytes=$(stat -f '%z' "$model_path" 2>/dev/null \
        || stat -c '%s' "$model_path")
    qwen38_validate_artifact_identity "$format" \
        "${artifact_sha256s[$artifact_index]}" "$current_bytes" "$file_type"
    artifact_index=$((artifact_index + 1))
done

[[ "$(git -C "$root_dir" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] || {
    echo "four-position artifact matrix source changed during execution" >&2
    exit 1
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
current_dependency_identity=$(qwen38_mlx_native_registry_identity "$root_dir")
[[ "$current_dependency_identity" == \
  "$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum" ]] || {
    echo "mlx-native dependency identity changed during matrix execution" >&2
    exit 1
}

matrix_results=$(jq -s . "${cell_paths[@]}")
runner_sha=$(shasum -a 256 \
    "$script_dir/qwen38_four_position_artifact_matrix.sh" | awk '{print $1}')
artifact_contract_sha=$(shasum -a 256 \
    "$script_dir/qwen38_artifact_contract.sh" | awk '{print $1}')
jq -n \
    --arg source_commit "$source_commit" \
    --arg dependency_version "$dependency_version" \
    --arg dependency_source "$dependency_source" \
    --arg dependency_checksum "$dependency_checksum" \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg runner_sha "$runner_sha" \
    --arg artifact_contract_sha "$artifact_contract_sha" \
    --argjson results "$matrix_results" '{
      schema:1,verdict:"pass",gate:"qwen38-four-position-artifact-matrix",
      source_commit:$source_commit,repository:$repository,revision:$revision,
      dependency:{name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum},
      evidence:{runner_sha256:$runner_sha,
        artifact_contract_sha256:$artifact_contract_sha},
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:$results
    }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_four_position_matrix_receipt "$OUT_DIR/matrix.json.tmp"

evidence_manifest="$OUT_DIR/evidence.sha256"
: >"$evidence_manifest.tmp"
while IFS= read -r path; do
    case "$path" in
        matrix.json|matrix.json.tmp|evidence.sha256|evidence.sha256.tmp|result.sha256|result.sha256.tmp)
            continue
            ;;
    esac
    printf '%s  %s\n' "$(shasum -a 256 "$OUT_DIR/$path" | awk '{print $1}')" \
        "$path" >>"$evidence_manifest.tmp"
done < <(cd "$OUT_DIR" && find . -type f -print | sed 's#^./##' | sort)
mv "$evidence_manifest.tmp" "$evidence_manifest"
(cd "$OUT_DIR" && shasum -a 256 -c evidence.sha256 >/dev/null)

matrix_sha=$(shasum -a 256 "$OUT_DIR/matrix.json.tmp" | awk '{print $1}')
evidence_sha=$(shasum -a 256 "$evidence_manifest" | awk '{print $1}')
printf '%s  matrix.json\n%s  evidence.sha256\n' "$matrix_sha" "$evidence_sha" \
    >"$OUT_DIR/result.sha256.tmp"
mv "$OUT_DIR/result.sha256.tmp" "$OUT_DIR/result.sha256"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
final_dependency_identity=$(qwen38_mlx_native_registry_identity "$root_dir") || {
    mv "$OUT_DIR/matrix.json" "$OUT_DIR/matrix.json.unsealed"
    exit 1
}
if ! qwen38_validate_four_position_matrix_seal "$OUT_DIR/matrix.json" "$root_dir" \
    || [[ "$(git -C "$root_dir" rev-parse HEAD)" != "$source_commit" ]] \
    || [[ -n "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] \
    || ! qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME" \
    || [[ "$final_dependency_identity" != \
      "$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum" ]]; then
    mv "$OUT_DIR/matrix.json" "$OUT_DIR/matrix.json.unsealed"
    exit 1
fi
jq . "$OUT_DIR/matrix.json"
