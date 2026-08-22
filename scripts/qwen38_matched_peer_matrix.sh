#!/usr/bin/env bash
set -euo pipefail

# Sequential ABBA comparison across every qualified Qwen3.8 storage format.
# The inner runner binds the exact peer commit from data/llama_cpp_pin.txt and
# refuses to publish a speed verdict unless quality, calibration, stability,
# and hf2q >= pinned-peer performance all pass for that format.

HF2Q_BIN=${HF2Q_BIN:?HF2Q_BIN is required}
HF2Q_COMMIT=${HF2Q_COMMIT:?HF2Q_COMMIT is required}
HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}
REFERENCE_BIN=${REFERENCE_BIN:?REFERENCE_BIN is required}
REFERENCE_SOURCE_DIR=${REFERENCE_SOURCE_DIR:?REFERENCE_SOURCE_DIR is required}
REFERENCE_COMMIT=${REFERENCE_COMMIT:?REFERENCE_COMMIT is required}
REFERENCE_SHA256=${REFERENCE_SHA256:?REFERENCE_SHA256 is required}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18086}

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
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "matched artifact-matrix output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
[[ -z ${HF2Q_MODEL_VERIFICATION_RECEIPT:-} ]] || {
    echo "one model-verification receipt cannot be shared across the artifact matrix" >&2
    exit 2
}
qwen38_validate_pinned_peer_commit "$REFERENCE_COMMIT"

# Refuse a partial catalog before the first calibrated model load.
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
    IFS=$'\t' read -r _format relative_file bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    format_out="$OUT_DIR/$format_slug"
    HF2Q_BIN="$HF2Q_BIN" \
    HF2Q_COMMIT="$HF2Q_COMMIT" \
    HF2Q_SHA256="$HF2Q_SHA256" \
    REFERENCE_BIN="$REFERENCE_BIN" \
    REFERENCE_SOURCE_DIR="$REFERENCE_SOURCE_DIR" \
    REFERENCE_COMMIT="$REFERENCE_COMMIT" \
    REFERENCE_SHA256="$REFERENCE_SHA256" \
    MODEL_FORMAT="$format" \
    MODEL_PATH="$MODEL_ROOT/$relative_file" \
    MODEL_SHA256="$sha256" \
    MODEL_BYTES="$bytes" \
    OUT_DIR="$format_out" \
    PORT="$PORT" \
        "$script_dir/qwen38_matched_reference_abba.sh"
    summary_paths+=("$format_out/summary.json")
done

matrix_results=$(jq -s . "${summary_paths[@]}")

jq -n \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg peer_commit "$(qwen38_pinned_peer_commit)" \
    --argjson results "$matrix_results" '{
      schema:1,verdict:"pass",gate:"qwen38-matched-peer-artifact-matrix",
      repository:$repository,revision:$revision,pinned_peer_commit:$peer_commit,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],results:$results
    }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_matched_peer_matrix_receipt "$OUT_DIR/matrix.json.tmp"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
jq . "$OUT_DIR/matrix.json"
