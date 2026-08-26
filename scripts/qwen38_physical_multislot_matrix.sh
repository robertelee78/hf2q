#!/usr/bin/env bash
set -euo pipefail

# Full qualified-artifact × physical-width proof. Each format is loaded in a
# fresh server and must pass N=1/2/4/8/16 with exact per-lane scalar replay.

BINARY_PATH=${BINARY_PATH:?BINARY_PATH is required}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
FOUR_POSITION_MATRIX_RECEIPT=${FOUR_POSITION_MATRIX_RECEIPT:?FOUR_POSITION_MATRIX_RECEIPT is required}
PORT=${PORT:-18092}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in awk find jq ps shasum stat tr; do
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
export HF2Q_MODEL_VERIFICATION_BINARY="$BINARY_PATH"
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "physical artifact-matrix output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
[[ -z ${HF2Q_MODEL_VERIFICATION_RECEIPT:-} ]] || {
    echo "one model-verification receipt cannot be shared across the artifact matrix" >&2
    exit 2
}
qwen38_validate_four_position_matrix_seal \
    "$FOUR_POSITION_MATRIX_RECEIPT" "$root_dir" || {
    echo "four-position route proof is invalid" >&2
    exit 2
}

# Preflight the complete catalog before loading the first model.  The v2
# recorder is the only full-artifact reader; its receipts are handed to the
# per-format children rather than independently rehashing every GGUF.
mkdir -p "$OUT_DIR/preflight"
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes expected_sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    [[ -f "$model_path" && -r "$model_path" ]] || {
        echo "qualified $format artifact is missing: $model_path" >&2
        exit 2
    }
    actual_bytes=$(stat -f '%z' "$model_path" 2>/dev/null \
        || stat -c '%s' "$model_path")
    receipt="$OUT_DIR/preflight/$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]').json"
    hf2q_release_prepare_model_verification "$model_path" "$expected_sha256" \
        "$receipt" "${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}/hf2q/model-verification}"
    qwen38_validate_artifact_identity \
        "$format" "$expected_sha256" "$actual_bytes" "$file_type"
done

mkdir -p "$OUT_DIR"
qwen38_copy_four_position_matrix_seal \
    "$FOUR_POSITION_MATRIX_RECEIPT" "$OUT_DIR/four-position" "$root_dir"
four_position_matrix_sha=$(shasum -a 256 \
    "$OUT_DIR/four-position/matrix.json" | awk '{print $1}')
matrix_child_pid=''

matrix_child_is_live() {
    local state
    kill -0 "$matrix_child_pid" 2>/dev/null || return 1
    state=$(ps -o stat= -p "$matrix_child_pid" 2>/dev/null \
        | awk 'NR == 1 {print $1}')
    [[ -n "$state" && "$state" != Z* ]]
}

terminate_matrix_child() {
    local signal=${1:-TERM} waited=0
    [[ -n "$matrix_child_pid" ]] || return 0
    if matrix_child_is_live; then
        kill -"$signal" "$matrix_child_pid" 2>/dev/null || true
        while matrix_child_is_live && ((waited < 60)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if matrix_child_is_live; then
        echo "physical matrix child ignored bounded shutdown; killing owned child $matrix_child_pid" >&2
        kill -KILL "$matrix_child_pid" 2>/dev/null || true
    fi
    wait "$matrix_child_pid" 2>/dev/null || true
    matrix_child_pid=''
}

on_matrix_exit() {
    local original_rc=$?
    trap - EXIT INT TERM
    terminate_matrix_child TERM
    exit "$original_rc"
}

on_matrix_signal() {
    local signal=$1 exit_code=$2
    trap - INT TERM
    terminate_matrix_child "$signal"
    exit "$exit_code"
}

trap on_matrix_exit EXIT
trap 'on_matrix_signal INT 130' INT
trap 'on_matrix_signal TERM 143' TERM

summary_paths=()
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes sha256 file_type \
        <<<"$(qwen38_artifact_record "$format")"
    format_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    format_out="$OUT_DIR/$format_slug"
    preflight_receipt="$OUT_DIR/preflight/$format_slug.json"
    BINARY_PATH="$BINARY_PATH" \
    MODEL_FORMAT="$format" \
    MODEL_PATH="$MODEL_ROOT/$relative_file" \
    MODEL_SHA256="$sha256" \
    HF2Q_MODEL_VERIFICATION_RECEIPT="$preflight_receipt" \
    OUT_DIR="$format_out" \
    PORT="$PORT" \
    MAX_TOKENS="$QWEN38_PHYSICAL_MAX_TOKENS" \
    KV_CACHE_BUDGET_BYTES="$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
    HF2Q_DECODE_MVN="$QWEN38_PHYSICAL_DECODE_MVN" \
    HF2Q_DECODE_MV_EXT="$QWEN38_PHYSICAL_DECODE_MV_EXT" \
    HF2Q_Q5K_CANONICAL_Q4X4="$QWEN38_PHYSICAL_Q5K_CANONICAL_Q4X4" \
        "$script_dir/qwen38_physical_multislot_gate.sh" &
    matrix_child_pid=$!
    child_rc=0
    wait "$matrix_child_pid" || child_rc=$?
    matrix_child_pid=''
    if ((child_rc != 0)); then
        echo "physical matrix child failed for $format with status $child_rc" >&2
        exit "$child_rc"
    fi
    summary_paths+=("$format_out/summary.json")
done

matrix_results=$(jq -s . "${summary_paths[@]}")
matrix_runner_sha=$(shasum -a 256 "$script_dir/qwen38_physical_multislot_matrix.sh" \
    | awk '{print $1}')
gate_runner_sha=$(shasum -a 256 "$script_dir/qwen38_physical_multislot_gate.sh" \
    | awk '{print $1}')
physical_contract_sha=$(shasum -a 256 \
    "$script_dir/qwen38_physical_multislot_contract.sh" | awk '{print $1}')
artifact_contract_sha=$(shasum -a 256 "$script_dir/qwen38_artifact_contract.sh" \
    | awk '{print $1}')

jq -n \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg matrix_runner_sha "$matrix_runner_sha" \
    --arg gate_runner_sha "$gate_runner_sha" \
    --arg physical_contract_sha "$physical_contract_sha" \
    --arg artifact_contract_sha "$artifact_contract_sha" \
    --argjson max_tokens "$QWEN38_PHYSICAL_MAX_TOKENS" \
    --argjson kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
    --argjson decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
    --argjson decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
    --arg four_position_matrix_sha "$four_position_matrix_sha" \
    --argjson results "$matrix_results" '{
      schema:2,verdict:"pass",gate:"qwen38-artifact-physical-width-matrix",
      repository:$repository,revision:$revision,
      formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
      widths:[1,2,4,8,16],
      workload:{max_tokens:$max_tokens,kv_cache_budget_bytes:$kv_budget,
        routing:{decode_mvn:$decode_mvn,decode_mv_ext:$decode_mv_ext,
          q5k_canonical_q4x4:true}},
      route_proof:{four_position_matrix_sha256:$four_position_matrix_sha,
        q5k:{qtype:"Q5_K",width:4,
          kernel:"kernel_mul_mv_ext_q5_K_f32_r1_4",actual_dispatch:true}},
      evidence:{matrix_runner_sha256:$matrix_runner_sha,
        gate_runner_sha256:$gate_runner_sha,
        physical_contract_sha256:$physical_contract_sha,
        artifact_contract_sha256:$artifact_contract_sha},
      results:$results
    }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_physical_matrix_receipt "$OUT_DIR/matrix.json.tmp"

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
if ! qwen38_validate_physical_matrix_seal "$OUT_DIR/matrix.json"; then
    mv "$OUT_DIR/matrix.json" "$OUT_DIR/matrix.json.unsealed"
    exit 1
fi
jq . "$OUT_DIR/matrix.json"
