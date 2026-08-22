#!/usr/bin/env bash
set -euo pipefail

# Full qualified-artifact × physical-width proof. Each format is loaded in a
# fresh server and must pass N=1/2/4/8/16 with exact per-lane scalar replay.

BINARY_PATH=${BINARY_PATH:?BINARY_PATH is required}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18092}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"

for command in awk find jq shasum stat tr; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$MODEL_ROOT" == /* && "$OUT_DIR" == /* ]] || {
    echo "MODEL_ROOT and OUT_DIR must be absolute paths" >&2
    exit 2
}
[[ -x "$BINARY_PATH" ]] || {
    echo "hf2q binary is missing or non-executable: $BINARY_PATH" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "physical artifact-matrix output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
[[ -z ${HF2Q_MODEL_VERIFICATION_RECEIPT:-} ]] || {
    echo "one model-verification receipt cannot be shared across the artifact matrix" >&2
    exit 2
}

# Preflight the complete catalog before loading the first model.
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes _expected_sha256 file_type \
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
done

mkdir -p "$OUT_DIR"
summary_paths=()
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    format_out="$OUT_DIR/$format_slug"
    BINARY_PATH="$BINARY_PATH" \
    MODEL_FORMAT="$format" \
    MODEL_PATH="$MODEL_ROOT/$relative_file" \
    MODEL_SHA256="$sha256" \
    OUT_DIR="$format_out" \
    PORT="$PORT" \
        "$script_dir/qwen38_physical_multislot_gate.sh"
    summary_paths+=("$format_out/summary.json")
done

matrix_results=$(jq -s . "${summary_paths[@]}")

jq -n \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --argjson results "$matrix_results" '{
      schema:1,verdict:"pass",gate:"qwen38-artifact-physical-width-matrix",
      repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
      widths:[1,2,4,8,16],results:$results
    }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_physical_matrix_receipt "$OUT_DIR/matrix.json.tmp"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
jq . "$OUT_DIR/matrix.json"
