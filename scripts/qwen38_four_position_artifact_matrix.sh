#!/usr/bin/env bash
set -euo pipefail

# Full-model native-storage and four-position trajectory proof for every
# qualified Qwen3.8 artifact. Each cargo test process unloads its model before
# the next format starts; the Q4_K_M cell also proves the qualified mv_ext
# width-four projection route.

MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"

for command in awk cargo find git jq mv sed shasum stat tr; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$MODEL_ROOT" == /* && "$OUT_DIR" == /* ]] || {
    echo "MODEL_ROOT and OUT_DIR must be absolute paths" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "four-position artifact-matrix output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
source_commit=$(git -C "$root_dir" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ \
    && -z "$(git -C "$root_dir" status --porcelain)" ]] || {
    echo "four-position artifact matrix requires a clean exact source tree" >&2
    exit 2
}

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
    artifact_snapshots+=("$(stat -f '%d:%i:%z:%m:%c' "$model_path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$model_path")")
done

mkdir -p "$OUT_DIR"
cell_paths=()
test_name='inference::models::qwen35::spec_decode::tests::qwen38_real_four_position_normal_forward_parity'
artifact_index=0
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file bytes expected_sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    actual_sha256=${artifact_sha256s[$artifact_index]}
    current_snapshot=$(stat -f '%d:%i:%z:%m:%c' "$model_path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$model_path")
    [[ "$current_snapshot" == "${artifact_snapshots[$artifact_index]}" ]] || {
        echo "$format artifact changed after matrix preflight" >&2
        exit 1
    }
    qwen38_validate_artifact_identity \
        "$format" "$actual_sha256" "$bytes" "$file_type"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    log_path="$OUT_DIR/$format_slug.log"
    if [[ "$format" == Q4_K_M ]]; then
        HF2Q_TEST_QWEN38_FORMAT="$format" \
        HF2Q_TEST_QWEN38_GGUF="$model_path" \
        HF2Q_TEST_QWEN38_EXPECT_MV_EXT=1 \
        MLX_DISP_BUCKET=1 \
        HF2Q_DECODE_MVN=0 \
        HF2Q_DECODE_MV_EXT=1 \
            cargo test --locked --bin hf2q "$test_name" \
                -- --ignored --exact --nocapture >"$log_path" 2>&1
    else
        env -u HF2Q_TEST_QWEN38_EXPECT_Q4K_MVN \
            -u HF2Q_TEST_QWEN38_EXPECT_MV_EXT \
            HF2Q_TEST_QWEN38_FORMAT="$format" \
            HF2Q_TEST_QWEN38_GGUF="$model_path" \
            cargo test --locked --bin hf2q "$test_name" \
                -- --ignored --exact --nocapture >"$log_path" 2>&1
    fi
    grep -Eq 'test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log_path" || {
        echo "$format did not execute exactly one passing full-model gate" >&2
        sed -n '1,260p' "$log_path" >&2
        exit 1
    }
    current_snapshot=$(stat -f '%d:%i:%z:%m:%c' "$model_path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$model_path")
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
        --argjson q4_mv_ext_proven "$([[ "$format" == Q4_K_M ]] && echo true || echo false)" '{
          schema:1,verdict:"pass",format:$format,
          artifact:{path:$model_path,sha256:$sha256,bytes:$bytes,
            gguf_file_type:$file_type},
          proof:{native_storage_without_substitution:true,
            four_position_matches_scalar:true,
            eight_token_post_batch_handoff_matches_scalar:true,
            q4_mv_ext_width_four:$q4_mv_ext_proven},
          log:{path:$log_path,sha256:$log_sha256}
        }' >"$cell_path.tmp"
    mv "$cell_path.tmp" "$cell_path"
    cell_paths+=("$cell_path")
    artifact_index=$((artifact_index + 1))
done

matrix_results=$(jq -s . "${cell_paths[@]}")
jq -n \
    --arg source_commit "$source_commit" \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --argjson results "$matrix_results" '{
      schema:1,verdict:"pass",gate:"qwen38-four-position-artifact-matrix",
      source_commit:$source_commit,repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:$results
    }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_four_position_matrix_receipt "$OUT_DIR/matrix.json.tmp"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
jq . "$OUT_DIR/matrix.json"
